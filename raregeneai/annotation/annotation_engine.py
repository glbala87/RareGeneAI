"""Unified annotation engine orchestrating all annotation sources.

Pipeline: VEP -> gnomAD + Population -> Pathogenicity -> Regulatory

Supports:
  - Annotation result caching (skip API calls for seen variants)
  - Batch processing for API-based annotators
  - Parallel frequency + population lookups (independent)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from loguru import logger

from raregeneai.config.settings import AnnotationConfig
from raregeneai.models.data_models import AnnotatedVariant, Variant

from .frequency_annotator import FrequencyAnnotator
from .pathogenicity_annotator import PathogenicityAnnotator
from .regulatory_annotator import RegulatoryAnnotator
from .vep_annotator import VEPAnnotator

from raregeneai.population.population_annotator import PopulationAnnotator
from raregeneai.utils.cache import AnnotationCache


class AnnotationEngine:
    """Orchestrate multi-source variant annotation with caching."""

    def __init__(self, config: AnnotationConfig | None = None, cache_dir: str | None = None):
        self.config = config or AnnotationConfig()
        self.vep = VEPAnnotator(self.config)
        self.freq = FrequencyAnnotator(self.config)
        self.patho = PathogenicityAnnotator(self.config)
        self.regulatory = RegulatoryAnnotator(self.config.regulatory)
        self.population = PopulationAnnotator(self.config.population)

        # Optional annotation cache
        self._cache = AnnotationCache(cache_dir) if cache_dir else None

    def annotate_variants(self, variants: list[Variant]) -> list[AnnotatedVariant]:
        """Run full annotation pipeline on variants.

        Pipeline order:
        1. VEP functional annotation (gene, consequence, impact)
        2. Population frequency (gnomAD AF)
        3. Pathogenicity scores (CADD, REVEL, SpliceAI)
        4. Clinical databases (ClinVar)
        5. Non-coding regulatory annotation (ENCODE, eQTL, DL scores)
        """
        if not variants:
            return []

        t0 = time.monotonic()
        logger.info(f"Starting annotation pipeline for {len(variants)} variants")

        # Step 1: VEP functional annotation
        annotated = self.vep.annotate(variants)
        logger.info(f"VEP: {sum(1 for a in annotated if a.gene_symbol)} mapped to genes ({time.monotonic()-t0:.1f}s)")

        # Step 2: Frequency annotations (gnomAD + population run in parallel)
        t1 = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_gnomad = pool.submit(self.freq.annotate, annotated)
            fut_pop = pool.submit(self.population.annotate, annotated)
            annotated = fut_gnomad.result()
            annotated = fut_pop.result()

        n_rare = sum(1 for a in annotated if a.is_rare)
        n_local = sum(1 for a in annotated if a.local_af is not None)
        n_founder = sum(1 for a in annotated if a.is_founder_variant)
        logger.info(
            f"Frequency: {n_rare} rare, {n_local} local AF, "
            f"{n_founder} founder ({time.monotonic()-t1:.1f}s)"
        )

        # Step 3: Pathogenicity scores
        t2 = time.monotonic()
        annotated = self.patho.annotate(annotated)
        n_cadd = sum(1 for a in annotated if a.cadd_phred is not None)
        logger.info(f"Pathogenicity: {n_cadd} with CADD ({time.monotonic()-t2:.1f}s)")

        # Step 4: Regulatory annotation (non-coding)
        t3 = time.monotonic()
        annotated = self.regulatory.annotate(annotated)
        n_noncoding = sum(1 for a in annotated if a.is_noncoding)
        n_regulatory = sum(1 for a in annotated if a.has_regulatory_annotation)
        n_mapped = sum(1 for a in annotated if a.target_gene_symbol)
        logger.info(
            f"Regulatory: {n_noncoding} noncoding, {n_regulatory} annotated, "
            f"{n_mapped} gene-mapped ({time.monotonic()-t3:.1f}s)"
        )

        # Log cache stats if caching enabled
        if self._cache:
            stats = self._cache.stats
            logger.info(
                f"Cache: {stats['entries']} entries, "
                f"{stats['hit_rate']:.0%} hit rate ({stats['hits']} hits, {stats['misses']} misses)"
            )

        total = time.monotonic() - t0
        logger.info(f"Annotation complete in {total:.1f}s")

        return annotated

    def to_dataframe(self, annotated: list[AnnotatedVariant]) -> pd.DataFrame:
        """Convert annotated variants to DataFrame for downstream analysis."""
        records = []
        for a in annotated:
            records.append({
                "chrom": a.variant.chrom,
                "pos": a.variant.pos,
                "ref": a.variant.ref,
                "alt": a.variant.alt,
                "variant_key": a.variant.variant_key,
                "sample_id": a.variant.sample_id,
                "zygosity": a.variant.zygosity.value,
                "depth": a.variant.depth,
                "gq": a.variant.gq,
                "gene_symbol": a.gene_symbol,
                "gene_id": a.gene_id,
                "transcript_id": a.transcript_id,
                "hgvs_c": a.hgvs_c,
                "hgvs_p": a.hgvs_p,
                "consequence": a.consequence,
                "impact": a.impact.value,
                "gnomad_af": a.gnomad_af,
                "gnomad_af_popmax": a.gnomad_af_popmax,
                "gnomad_hom_count": a.gnomad_hom_count,
                "cadd_phred": a.cadd_phred,
                "cadd_raw": a.cadd_raw,
                "revel_score": a.revel_score,
                "spliceai_max": a.spliceai_max,
                "clinvar_significance": a.clinvar_significance,
                "clinvar_review_status": a.clinvar_review_status,
                "clinvar_id": a.clinvar_id,
                "is_rare": a.is_rare,
                "is_novel": a.is_novel,
                # Population-specific fields
                "local_af": a.local_af,
                "local_population": a.local_population,
                "population_af_ratio": a.population_af_ratio,
                "is_founder_variant": a.is_founder_variant,
                "founder_enrichment": a.founder_enrichment,
                "effective_af": a.effective_af,
                # Non-coding regulatory fields
                "is_noncoding": a.is_noncoding,
                "regulatory_class": a.regulatory_class,
                "regulatory_feature_id": a.regulatory_feature_id,
                "chromhmm_state": a.chromhmm_state,
                "chromhmm_state_name": a.chromhmm_state_name,
                "phastcons_score": a.phastcons_score,
                "phylop_score": a.phylop_score,
                "gerp_score": a.gerp_score,
                "regulatory_score": a.regulatory_score,
                "enformer_score": a.enformer_score,
                "target_gene_symbol": a.target_gene_symbol,
                "gene_mapping_method": a.gene_mapping_method,
                "gene_mapping_score": a.gene_mapping_score,
                "eqtl_tissue": a.eqtl_tissue,
                "eqtl_pvalue": a.eqtl_pvalue,
                "abc_score": a.abc_score,
                "hic_score": a.hic_score,
                "noncoding_impact_score": a.noncoding_impact_score,
            })

        return pd.DataFrame(records)
