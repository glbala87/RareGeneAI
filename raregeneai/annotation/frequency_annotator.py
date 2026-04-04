"""Population frequency annotation from gnomAD and other databases.

Supports local gnomAD VCF/database or remote API queries.
"""

from __future__ import annotations

from pathlib import Path

import requests
from loguru import logger

from raregeneai.config.settings import AnnotationConfig
from raregeneai.models.data_models import AnnotatedVariant


class FrequencyAnnotator:
    """Annotate variants with population allele frequencies."""

    def __init__(self, config: AnnotationConfig | None = None):
        self.config = config or AnnotationConfig()
        self._gnomad_tabix = None

    def annotate(self, variants: list[AnnotatedVariant]) -> list[AnnotatedVariant]:
        """Add gnomAD allele frequency annotations."""
        if not variants:
            return variants

        logger.info(f"Annotating {len(variants)} variants with population frequencies")

        if self.config.gnomad_path and Path(self.config.gnomad_path).exists():
            return self._annotate_local(variants)
        return self._annotate_remote(variants)

    def _annotate_local(self, variants: list[AnnotatedVariant]) -> list[AnnotatedVariant]:
        """Query local gnomAD VCF using tabix/pysam."""
        try:
            import pysam

            if self._gnomad_tabix is None:
                self._gnomad_tabix = pysam.TabixFile(self.config.gnomad_path)

            for var in variants:
                try:
                    region = f"{var.variant.chrom}:{var.variant.pos}-{var.variant.pos}"
                    for row in self._gnomad_tabix.fetch(region=region):
                        fields = row.split("\t")
                        if len(fields) < 8:
                            continue
                        ref, alt = fields[3], fields[4]
                        if ref == var.variant.ref and alt == var.variant.alt:
                            info = fields[7]
                            var.gnomad_af = self._parse_af_from_info(info, "AF")
                            var.gnomad_af_popmax = self._parse_af_from_info(info, "AF_popmax")
                            var.gnomad_hom_count = int(
                                self._parse_af_from_info(info, "nhomalt") or 0
                            )
                            break
                except Exception:
                    pass

        except ImportError:
            logger.warning("pysam not available for local gnomAD query")

        return variants

    def _annotate_remote(self, variants: list[AnnotatedVariant]) -> list[AnnotatedVariant]:
        """Query gnomAD GraphQL API for allele frequencies."""
        for var in variants:
            try:
                af_data = self._query_gnomad_api(var)
                if af_data:
                    var.gnomad_af = af_data.get("af")
                    var.gnomad_af_popmax = af_data.get("af_popmax")
                    var.gnomad_hom_count = af_data.get("hom_count", 0)
            except Exception as e:
                logger.debug(f"gnomAD API query failed for {var.variant.variant_key}: {e}")

        return variants

    def _query_gnomad_api(self, var: AnnotatedVariant) -> dict | None:
        """Query gnomAD GraphQL endpoint."""
        query = """
        query GnomadVariant($variantId: String!, $dataset: DatasetId!) {
          variant(variantId: $variantId, dataset: $dataset) {
            exome {
              ac
              an
              homozygote_count
              populations {
                id
                ac
                an
              }
            }
            genome {
              ac
              an
              homozygote_count
              populations {
                id
                ac
                an
              }
            }
          }
        }
        """

        variant_id = f"{var.variant.chrom}-{var.variant.pos}-{var.variant.ref}-{var.variant.alt}"

        try:
            resp = requests.post(
                "https://gnomad.broadinstitute.org/api",
                json={
                    "query": query,
                    "variables": {
                        "variantId": variant_id,
                        "dataset": "gnomad_r4",
                    },
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            variant_data = data.get("data", {}).get("variant")
            if not variant_data:
                return None

            # Combine exome and genome data
            total_ac = 0
            total_an = 0
            total_hom = 0
            max_pop_af = 0.0

            for source in ["exome", "genome"]:
                src_data = variant_data.get(source)
                if src_data:
                    total_ac += src_data.get("ac", 0) or 0
                    total_an += src_data.get("an", 0) or 0
                    total_hom += src_data.get("homozygote_count", 0) or 0

                    for pop in src_data.get("populations", []):
                        pop_an = pop.get("an", 0) or 0
                        pop_ac = pop.get("ac", 0) or 0
                        if pop_an > 0:
                            pop_af = pop_ac / pop_an
                            max_pop_af = max(max_pop_af, pop_af)

            af = total_ac / total_an if total_an > 0 else None

            return {
                "af": af,
                "af_popmax": max_pop_af if max_pop_af > 0 else af,
                "hom_count": total_hom,
            }

        except Exception:
            return None

    @staticmethod
    def _parse_af_from_info(info: str, key: str) -> float | None:
        for field in info.split(";"):
            if field.startswith(f"{key}="):
                try:
                    return float(field.split("=")[1])
                except ValueError:
                    return None
        return None
