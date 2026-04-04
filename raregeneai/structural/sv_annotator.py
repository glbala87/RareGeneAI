"""Structural variant annotation and scoring engine.

Annotates SVs with:
  1. Gene overlap (full deletion, partial overlap, exon disruption)
  2. AnnotSV classification (pathogenicity ranking)
  3. Dosage sensitivity (pLI, LOEUF, ClinGen HI/TS)
  4. Population frequency (gnomAD-SV, DGV)
  5. Regulatory disruption (TAD boundaries, enhancers, promoters)
  6. Clinical databases (ClinVar SVs, DECIPHER)

Produces a composite SV pathogenicity score per variant and per gene.
"""

from __future__ import annotations

import csv
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from raregeneai.config.settings import SVConfig
from raregeneai.models.data_models import (
    AnnotatedSV,
    StructuralVariant,
    SVType,
)


# ── Dosage sensitivity thresholds ─────────────────────────────────────────────
PLI_INTOLERANT = 0.9        # Gene is LoF-intolerant
LOEUF_CONSTRAINED = 0.35    # Gene is constrained
CLINGEN_HI_SUFFICIENT = 3   # ClinGen HI score indicating sufficient evidence


class SVAnnotator:
    """Annotate structural variants with gene overlap, dosage, and regulatory data."""

    def __init__(self, config: SVConfig | None = None):
        self.config = config or SVConfig()
        self._pli_scores: dict[str, float] = {}
        self._loeuf_scores: dict[str, float] = {}
        self._hi_scores: dict[str, float] = {}
        self._ts_scores: dict[str, float] = {}
        self._gene_coords: dict[str, tuple[str, int, int]] = {}
        self._loaded = False

    def annotate(self, svs: list[StructuralVariant]) -> list[AnnotatedSV]:
        """Run the full SV annotation pipeline.

        Pipeline:
        1. Gene overlap analysis
        2. AnnotSV classification (if available)
        3. Dosage sensitivity lookup
        4. Population frequency
        5. Regulatory disruption
        6. Clinical database lookup
        7. Composite score computation
        """
        if not svs:
            return []

        self._load_reference_data()
        logger.info(f"Annotating {len(svs)} structural variants")

        annotated = []
        for sv in svs:
            ann = AnnotatedSV(sv=sv)

            self._annotate_gene_overlap(ann)
            self._annotate_dosage_sensitivity(ann)
            self._annotate_population_frequency(ann)
            self._annotate_regulatory_disruption(ann)
            self._annotate_clinical_databases(ann)
            self._compute_sv_scores(ann)

            annotated.append(ann)

        # Try AnnotSV batch annotation if available
        self._run_annotsv_batch(annotated)

        n_with_genes = sum(1 for a in annotated if a.overlapping_genes)
        n_rare = sum(1 for a in annotated if a.is_rare)
        n_dosage = sum(1 for a in annotated if a.has_dosage_sensitive_gene)
        logger.info(
            f"SV annotation complete: {n_with_genes} overlap genes, "
            f"{n_rare} rare, {n_dosage} affect dosage-sensitive genes"
        )

        return annotated

    def _load_reference_data(self) -> None:
        """Load gene coordinates, pLI/LOEUF scores, ClinGen dosage data."""
        if self._loaded:
            return

        self._load_gene_coordinates()
        self._load_dosage_scores()
        self._loaded = True

    def _load_gene_coordinates(self) -> None:
        """Load gene BED file: chrom<TAB>start<TAB>end<TAB>gene_symbol."""
        if not (self.config.gene_bed_path and Path(self.config.gene_bed_path).exists()):
            return

        with open(self.config.gene_bed_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    gene = parts[3]
                    self._gene_coords[gene] = (parts[0], int(parts[1]), int(parts[2]))

        logger.info(f"Loaded coordinates for {len(self._gene_coords)} genes")

    def _load_dosage_scores(self) -> None:
        """Load pLI, LOEUF, ClinGen HI/TS scores."""
        # pLI scores (gene<TAB>pLI)
        if self.config.pli_scores_path and Path(self.config.pli_scores_path).exists():
            with open(self.config.pli_scores_path) as f:
                for line in f:
                    if line.startswith("#") or line.startswith("gene"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        try:
                            self._pli_scores[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
            logger.info(f"Loaded pLI scores for {len(self._pli_scores)} genes")

        # LOEUF scores
        if self.config.loeuf_scores_path and Path(self.config.loeuf_scores_path).exists():
            with open(self.config.loeuf_scores_path) as f:
                for line in f:
                    if line.startswith("#") or line.startswith("gene"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        try:
                            self._loeuf_scores[parts[0]] = float(parts[1])
                        except ValueError:
                            pass

        # ClinGen HI scores
        if self.config.clingen_hi_path and Path(self.config.clingen_hi_path).exists():
            with open(self.config.clingen_hi_path) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    gene = row.get("Gene Symbol", row.get("gene", ""))
                    score = row.get("Haploinsufficiency Score", row.get("hi_score", ""))
                    if gene and score:
                        try:
                            self._hi_scores[gene] = float(score)
                        except ValueError:
                            pass

        # ClinGen TS scores
        if self.config.clingen_ts_path and Path(self.config.clingen_ts_path).exists():
            with open(self.config.clingen_ts_path) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    gene = row.get("Gene Symbol", row.get("gene", ""))
                    score = row.get("Triplosensitivity Score", row.get("ts_score", ""))
                    if gene and score:
                        try:
                            self._ts_scores[gene] = float(score)
                        except ValueError:
                            pass

    # ── 1. Gene Overlap ────────────────────────────────────────────────────────

    def _annotate_gene_overlap(self, ann: AnnotatedSV) -> None:
        """Determine which genes are overlapped by the SV.

        Categories:
          - fully_deleted: Gene entirely within DEL boundaries
          - partially_overlapping: Partial overlap
          - overlapping_genes: Union of both
        """
        sv = ann.sv

        if sv.sv_type == SVType.BND:
            # BND: check both breakpoints
            self._annotate_bnd_overlap(ann)
            return

        sv_chrom = sv.chrom
        sv_start = sv.pos
        sv_end = sv.end if sv.end > sv.pos else sv.pos + sv.sv_len

        for gene, (g_chrom, g_start, g_end) in self._gene_coords.items():
            if g_chrom != sv_chrom:
                continue

            # Check overlap
            if sv_start <= g_end and sv_end >= g_start:
                ann.overlapping_genes.append(gene)

                # Full containment (gene entirely within SV)
                if sv_start <= g_start and sv_end >= g_end:
                    ann.fully_deleted_genes.append(gene)
                else:
                    ann.partially_overlapping_genes.append(gene)

        # If no local gene coords loaded, try to use Ensembl API
        if not self._gene_coords and not ann.overlapping_genes:
            self._gene_overlap_remote(ann)

    def _annotate_bnd_overlap(self, ann: AnnotatedSV) -> None:
        """For BND/translocations, check both breakpoints for gene disruption."""
        sv = ann.sv
        for gene, (g_chrom, g_start, g_end) in self._gene_coords.items():
            # Check breakpoint 1
            if g_chrom == sv.chrom and g_start <= sv.pos <= g_end:
                ann.overlapping_genes.append(gene)
                ann.partially_overlapping_genes.append(gene)
            # Check breakpoint 2
            if sv.chrom2 and g_chrom == sv.chrom2 and g_start <= sv.pos2 <= g_end:
                if gene not in ann.overlapping_genes:
                    ann.overlapping_genes.append(gene)
                    ann.partially_overlapping_genes.append(gene)

    def _gene_overlap_remote(self, ann: AnnotatedSV) -> None:
        """Query Ensembl REST API for gene overlap."""
        sv = ann.sv
        try:
            chrom = sv.chrom.replace("chr", "")
            end = sv.end if sv.end > sv.pos else sv.pos + sv.sv_len
            url = (
                f"https://rest.ensembl.org/overlap/region/human/"
                f"{chrom}:{sv.pos}-{end}?feature=gene;content-type=application/json"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                for gene_data in resp.json():
                    symbol = gene_data.get("external_name", "")
                    if symbol and gene_data.get("biotype") == "protein_coding":
                        ann.overlapping_genes.append(symbol)
                        g_start = gene_data.get("start", 0)
                        g_end_pos = gene_data.get("end", 0)
                        if sv.pos <= g_start and end >= g_end_pos:
                            ann.fully_deleted_genes.append(symbol)
                        else:
                            ann.partially_overlapping_genes.append(symbol)
        except Exception as e:
            logger.debug(f"Ensembl gene overlap query failed: {e}")

    # ── 2. Dosage Sensitivity ──────────────────────────────────────────────────

    def _annotate_dosage_sensitivity(self, ann: AnnotatedSV) -> None:
        """Look up dosage sensitivity scores for overlapping genes.

        Uses the most sensitive gene to set the SV-level scores.
        """
        max_pli = 0.0
        min_loeuf = 2.0
        max_hi = 0.0
        max_ts = 0.0

        for gene in ann.overlapping_genes:
            pli = self._pli_scores.get(gene, 0.0)
            loeuf = self._loeuf_scores.get(gene, 2.0)
            hi = self._hi_scores.get(gene, 0.0)
            ts = self._ts_scores.get(gene, 0.0)

            max_pli = max(max_pli, pli)
            min_loeuf = min(min_loeuf, loeuf)
            max_hi = max(max_hi, hi)
            max_ts = max(max_ts, ts)

        ann.pli_score = max_pli if max_pli > 0 else None
        ann.loeuf_score = min_loeuf if min_loeuf < 2.0 else None
        ann.hi_score = max_hi if max_hi > 0 else None
        ann.ts_score = max_ts if max_ts > 0 else None

    # ── 3. Population Frequency ────────────────────────────────────────────────

    def _annotate_population_frequency(self, ann: AnnotatedSV) -> None:
        """Query gnomAD-SV and DGV for population frequency."""
        sv = ann.sv

        # Local gnomAD-SV query
        if self.config.gnomad_sv_path and Path(self.config.gnomad_sv_path).exists():
            self._query_gnomad_sv_local(ann)
        else:
            self._query_gnomad_sv_remote(ann)

        # Local DGV query
        if self.config.dgv_path and Path(self.config.dgv_path).exists():
            self._query_dgv_local(ann)

    def _query_gnomad_sv_local(self, ann: AnnotatedSV) -> None:
        """Query local gnomAD-SV VCF via tabix."""
        try:
            import pysam

            sv = ann.sv
            tbx = pysam.TabixFile(self.config.gnomad_sv_path)
            end = sv.end if sv.end > sv.pos else sv.pos + sv.sv_len

            for row in tbx.fetch(sv.chrom, sv.pos - 1, end):
                fields = row.split("\t")
                if len(fields) < 8:
                    continue
                info = fields[7]
                # Check SVTYPE match
                for item in info.split(";"):
                    if item.startswith("SVTYPE="):
                        if item.split("=")[1] == sv.sv_type.value:
                            # Extract AF
                            for item2 in info.split(";"):
                                if item2.startswith("AF="):
                                    try:
                                        ann.gnomad_sv_af = float(item2.split("=")[1])
                                    except ValueError:
                                        pass
                                    return
        except Exception as e:
            logger.debug(f"gnomAD-SV local query failed: {e}")

    def _query_gnomad_sv_remote(self, ann: AnnotatedSV) -> None:
        """Estimate SV frequency using gnomAD API (simplified)."""
        # gnomAD doesn't have a direct SV GraphQL endpoint for arbitrary queries.
        # In production, use local gnomAD-SV VCF.
        pass

    def _query_dgv_local(self, ann: AnnotatedSV) -> None:
        """Query local DGV database."""
        try:
            import pysam

            sv = ann.sv
            tbx = pysam.TabixFile(self.config.dgv_path)
            end = sv.end if sv.end > sv.pos else sv.pos + sv.sv_len

            for row in tbx.fetch(sv.chrom, sv.pos - 1, end):
                fields = row.split("\t")
                if len(fields) >= 5:
                    try:
                        ann.dgv_frequency = float(fields[4])
                        return
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug(f"DGV query failed: {e}")

    # ── 4. Regulatory Disruption ───────────────────────────────────────────────

    def _annotate_regulatory_disruption(self, ann: AnnotatedSV) -> None:
        """Assess regulatory element disruption by the SV.

        Counts disrupted enhancers, promoters, CTCF sites, and TAD boundaries.
        """
        sv = ann.sv
        sv_start = sv.pos
        sv_end = sv.end if sv.end > sv.pos else sv.pos + sv.sv_len

        # TAD boundary disruption
        if self.config.tad_bed_path and Path(self.config.tad_bed_path).exists():
            ann.disrupted_tads = self._count_bed_overlaps(
                self.config.tad_bed_path, sv.chrom, sv_start, sv_end,
                return_names=True
            )

        # Regulatory element disruption
        if self.config.regulatory_bed_path and Path(self.config.regulatory_bed_path).exists():
            self._count_regulatory_overlaps(ann, sv.chrom, sv_start, sv_end)
        else:
            # Estimate from SV size and region
            self._estimate_regulatory_disruption(ann)

        # Compute disruption score
        disruption_score = 0.0
        if ann.disrupted_tads:
            disruption_score += 0.3 * min(len(ann.disrupted_tads), 3) / 3.0
        if ann.disrupted_enhancers > 0:
            disruption_score += 0.3 * min(ann.disrupted_enhancers, 5) / 5.0
        if ann.disrupted_promoters > 0:
            disruption_score += 0.25 * min(ann.disrupted_promoters, 3) / 3.0
        if ann.disrupted_ctcf_sites > 0:
            disruption_score += 0.15 * min(ann.disrupted_ctcf_sites, 5) / 5.0

        ann.regulatory_disruption_score = min(disruption_score, 1.0)

    def _count_bed_overlaps(
        self, bed_path: str, chrom: str, start: int, end: int,
        return_names: bool = False,
    ) -> list[str]:
        """Count overlaps with a BED file using tabix."""
        results = []
        try:
            import pysam
            tbx = pysam.TabixFile(bed_path)
            for row in tbx.fetch(chrom, start, end):
                fields = row.split("\t")
                if return_names and len(fields) >= 4:
                    results.append(fields[3])
                else:
                    results.append(f"{fields[0]}:{fields[1]}-{fields[2]}")
        except Exception as e:
            logger.debug(f"BED overlap query failed for {bed_path}: {e}")
        return results

    def _count_regulatory_overlaps(
        self, ann: AnnotatedSV, chrom: str, start: int, end: int,
    ) -> None:
        """Count disrupted regulatory elements from a BED with type column.

        BED format: chrom<TAB>start<TAB>end<TAB>name<TAB>type
        Types: enhancer, promoter, CTCF, open_chromatin
        """
        try:
            import pysam
            tbx = pysam.TabixFile(self.config.regulatory_bed_path)
            for row in tbx.fetch(chrom, start, end):
                fields = row.split("\t")
                if len(fields) >= 5:
                    elem_type = fields[4].lower()
                    if "enhancer" in elem_type:
                        ann.disrupted_enhancers += 1
                    elif "promoter" in elem_type:
                        ann.disrupted_promoters += 1
                    elif "ctcf" in elem_type or "insulator" in elem_type:
                        ann.disrupted_ctcf_sites += 1
        except Exception as e:
            logger.debug(f"Regulatory BED query failed: {e}")

    def _estimate_regulatory_disruption(self, ann: AnnotatedSV) -> None:
        """Heuristic: estimate regulatory disruption from SV size."""
        sv = ann.sv
        if sv.sv_type == SVType.BND:
            ann.disrupted_enhancers = 1  # BND likely disrupts at least one element
            return

        # Larger SVs disrupt more elements (rough estimate)
        if sv.sv_len > 500_000:
            ann.disrupted_enhancers = max(ann.disrupted_enhancers, 5)
            ann.disrupted_promoters = max(ann.disrupted_promoters, 2)
            ann.disrupted_ctcf_sites = max(ann.disrupted_ctcf_sites, 3)
        elif sv.sv_len > 100_000:
            ann.disrupted_enhancers = max(ann.disrupted_enhancers, 3)
            ann.disrupted_promoters = max(ann.disrupted_promoters, 1)
            ann.disrupted_ctcf_sites = max(ann.disrupted_ctcf_sites, 1)
        elif sv.sv_len > 10_000:
            ann.disrupted_enhancers = max(ann.disrupted_enhancers, 1)

    # ── 5. Clinical Databases ──────────────────────────────────────────────────

    def _annotate_clinical_databases(self, ann: AnnotatedSV) -> None:
        """Check ClinVar SVs and DECIPHER for known pathogenic SVs."""
        if self.config.clinvar_sv_path and Path(self.config.clinvar_sv_path).exists():
            self._query_clinvar_sv(ann)

        if self.config.decipher_path and Path(self.config.decipher_path).exists():
            self._query_decipher(ann)

    def _query_clinvar_sv(self, ann: AnnotatedSV) -> None:
        """Query local ClinVar SV database."""
        try:
            import pysam
            sv = ann.sv
            tbx = pysam.TabixFile(self.config.clinvar_sv_path)
            end = sv.end if sv.end > sv.pos else sv.pos + sv.sv_len

            for row in tbx.fetch(sv.chrom, sv.pos - 1, end):
                fields = row.split("\t")
                if len(fields) >= 8:
                    info = fields[7]
                    for item in info.split(";"):
                        if item.startswith("CLNSIG="):
                            ann.clinvar_sv_significance = item.split("=")[1]
                            return
        except Exception as e:
            logger.debug(f"ClinVar SV query failed: {e}")

    def _query_decipher(self, ann: AnnotatedSV) -> None:
        """Query DECIPHER CNV syndrome database."""
        try:
            import pysam
            sv = ann.sv
            tbx = pysam.TabixFile(self.config.decipher_path)
            end = sv.end if sv.end > sv.pos else sv.pos + sv.sv_len

            for row in tbx.fetch(sv.chrom, sv.pos - 1, end):
                fields = row.split("\t")
                if len(fields) >= 4:
                    ann.known_syndrome = fields[3]
                    if len(fields) >= 5:
                        ann.decipher_id = fields[4]
                    return
        except Exception as e:
            logger.debug(f"DECIPHER query failed: {e}")

    # ── 6. AnnotSV Batch Processing ───────────────────────────────────────────

    def _run_annotsv_batch(self, annotated: list[AnnotatedSV]) -> None:
        """Run AnnotSV on all SVs for comprehensive annotation.

        AnnotSV provides integrated annotations including:
        - Gene overlap with functional impact
        - Pathogenicity ranking (1-5)
        - Known CNV syndrome matching
        - DGV / gnomAD-SV frequency
        """
        try:
            # Check if AnnotSV is available
            result = subprocess.run(
                [self.config.annotsv_executable, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                logger.debug("AnnotSV not available, using built-in annotations")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("AnnotSV not found, using built-in annotations")
            return

        # Write SVs to temp BED for AnnotSV
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bed", delete=False
        ) as tmp:
            for ann in annotated:
                sv = ann.sv
                end = sv.end if sv.end > sv.pos else sv.pos + sv.sv_len
                tmp.write(f"{sv.chrom}\t{sv.pos}\t{end}\t{sv.sv_type.value}\t{sv.sv_len}\n")
            tmp_bed = tmp.name

        output_path = tmp_bed + ".annotsv.tsv"

        cmd = [
            self.config.annotsv_executable,
            "-SVinputFile", tmp_bed,
            "-outputFile", output_path,
            "-genomeBuild", "GRCh38",
            "-SVinputInfo", "1",
        ]

        if self.config.annotsv_annotations_dir:
            cmd.extend(["-annotationsDir", self.config.annotsv_annotations_dir])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)

            # Parse AnnotSV output
            if Path(output_path).exists():
                self._parse_annotsv_output(output_path, annotated)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"AnnotSV execution failed: {e}")
        finally:
            Path(tmp_bed).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def _parse_annotsv_output(self, output_path: str, annotated: list[AnnotatedSV]) -> None:
        """Parse AnnotSV TSV output and merge into AnnotatedSV objects."""
        try:
            with open(output_path) as f:
                reader = csv.DictReader(f, delimiter="\t")
                results = list(reader)

            for i, ann in enumerate(annotated):
                if i >= len(results):
                    break
                row = results[i]

                # AnnotSV class (1-5, where 5 = pathogenic)
                annotsv_class_str = row.get("AnnotSV_ranking_score", "")
                if annotsv_class_str:
                    try:
                        ann.annotsv_rank = int(float(annotsv_class_str))
                    except ValueError:
                        pass

                annotsv_classification = row.get("AnnotSV_ranking_criteria", "")
                if annotsv_classification:
                    ann.annotsv_class = annotsv_classification

                # Gene info
                gene = row.get("Gene_name", "")
                if gene and gene not in ann.overlapping_genes:
                    ann.overlapping_genes.append(gene)

        except Exception as e:
            logger.debug(f"AnnotSV output parsing failed: {e}")

    # ── 7. Composite Score Computation ─────────────────────────────────────────

    def _compute_sv_scores(self, ann: AnnotatedSV) -> None:
        """Compute all SV sub-scores and final composite score.

        Formula:
          sv_composite = w_overlap * gene_overlap_score
                       + w_dosage * dosage_sensitivity_score
                       + w_rarity * rarity_score
                       + w_regulatory * regulatory_disruption_score
                       + w_clinical * clinical_score
        """
        cfg = self.config
        sv = ann.sv

        # ── Gene overlap score ────────────────────────────────────────────
        overlap_score = 0.0
        if ann.fully_deleted_genes:
            # Full gene deletion is very high impact
            overlap_score = min(0.5 + 0.1 * len(ann.fully_deleted_genes), 1.0)
        elif ann.partially_overlapping_genes:
            # Partial overlap: depends on SV type
            if sv.is_deletion:
                overlap_score = 0.5 + 0.05 * len(ann.partially_overlapping_genes)
            elif sv.is_duplication:
                overlap_score = 0.3 + 0.05 * len(ann.partially_overlapping_genes)
            elif sv.sv_type == SVType.INV:
                overlap_score = 0.4 + 0.05 * len(ann.partially_overlapping_genes)
            elif sv.sv_type == SVType.BND:
                overlap_score = 0.6  # Translocations disrupting genes are severe
            else:
                overlap_score = 0.3

        overlap_score = min(overlap_score, 1.0)
        ann.gene_overlap_score = overlap_score

        # ── Dosage sensitivity score ──────────────────────────────────────
        dosage_score = 0.0
        if sv.is_deletion:
            # Deletion: haploinsufficiency matters
            if ann.pli_score is not None and ann.pli_score > PLI_INTOLERANT:
                dosage_score = max(dosage_score, ann.pli_score)
            if ann.hi_score is not None and ann.hi_score >= CLINGEN_HI_SUFFICIENT:
                dosage_score = max(dosage_score, 0.95)
            if ann.loeuf_score is not None and ann.loeuf_score < LOEUF_CONSTRAINED:
                dosage_score = max(dosage_score, 1.0 - ann.loeuf_score)
        elif sv.is_duplication:
            # Duplication: triplosensitivity matters
            if ann.ts_score is not None and ann.ts_score >= CLINGEN_HI_SUFFICIENT:
                dosage_score = max(dosage_score, 0.90)
            elif ann.pli_score is not None and ann.pli_score > PLI_INTOLERANT:
                # Dosage-sensitive genes are also sensitive to gain
                dosage_score = max(dosage_score, ann.pli_score * 0.6)

        dosage_score = min(dosage_score, 1.0)
        ann.dosage_sensitivity_score = dosage_score

        # ── Rarity score ──────────────────────────────────────────────────
        rarity_score = 0.0
        af = ann.gnomad_sv_af or ann.dgv_frequency
        if af is None:
            rarity_score = 1.0  # Novel SV
        elif af == 0.0:
            rarity_score = 1.0
        elif af < cfg.gnomad_sv_af_threshold:
            rarity_score = math.exp(-100 * af)
        else:
            rarity_score = 0.0  # Common SV
        ann.sv_rarity_score = rarity_score

        # ── Clinical score ────────────────────────────────────────────────
        clinical_score = 0.0
        clinvar_lower = ann.clinvar_sv_significance.lower()
        if "pathogenic" in clinvar_lower and "likely" not in clinvar_lower:
            clinical_score = 1.0
        elif "likely pathogenic" in clinvar_lower or "likely_pathogenic" in clinvar_lower:
            clinical_score = 0.85
        if ann.known_syndrome:
            clinical_score = max(clinical_score, 0.90)
        if ann.annotsv_rank >= 4:
            clinical_score = max(clinical_score, 0.80)
        elif ann.annotsv_rank == 3:
            clinical_score = max(clinical_score, 0.50)

        # ── SV pathogenicity (raw score) ──────────────────────────────────
        ann.sv_pathogenicity_score = max(
            overlap_score * 0.5 + dosage_score * 0.5,
            clinical_score,
        )

        # ── Composite ─────────────────────────────────────────────────────
        ann.sv_composite_score = (
            cfg.w_sv_gene_overlap * overlap_score
            + cfg.w_sv_dosage * dosage_score
            + cfg.w_sv_rarity * rarity_score
            + cfg.w_sv_regulatory * ann.regulatory_disruption_score
            + cfg.w_sv_clinical * clinical_score
        )
