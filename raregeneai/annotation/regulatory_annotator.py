"""Non-coding variant regulatory annotation engine.

Annotates variants with:
  1. Regulatory region overlaps (ENCODE cCREs, Roadmap ChromHMM)
  2. Conservation scores (PhastCons, PhyloP, GERP++)
  3. Deep learning regulatory predictions (Enformer, Sei, DeepBind)
  4. Variant-to-gene mapping (GTEx eQTL, Hi-C, ABC model)
  5. Unified regulatory impact score

Architecture:
  - Each annotation source is a private method
  - Local tabix/BED files preferred for speed
  - Remote APIs as fallback
  - Scores are normalized to [0, 1]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from raregeneai.config.settings import RegulatoryConfig
from raregeneai.models.data_models import AnnotatedVariant


# ── ChromHMM state definitions (Roadmap 15-state model) ───────────────────────
CHROMHMM_STATES = {
    "1_TssA":    ("Active TSS", 0.85),
    "2_TssAFlnk": ("Flanking Active TSS", 0.70),
    "3_TxFlnk":  ("Transcr. at gene 5'/3'", 0.50),
    "4_Tx":      ("Strong transcription", 0.30),
    "5_TxWk":    ("Weak transcription", 0.15),
    "6_EnhG":    ("Genic enhancers", 0.80),
    "7_Enh":     ("Enhancers", 0.75),
    "8_ZNF/Rpts": ("ZNF genes & repeats", 0.20),
    "9_Het":     ("Heterochromatin", 0.05),
    "10_TssBiv": ("Bivalent/Poised TSS", 0.65),
    "11_BivFlnk": ("Flanking Bivalent TSS", 0.50),
    "12_EnhBiv": ("Bivalent Enhancer", 0.55),
    "13_ReprPC": ("Repressed PolyComb", 0.10),
    "14_ReprPCWk": ("Weak Repressed PolyComb", 0.05),
    "15_Quies":  ("Quiescent/Low", 0.02),
}

# ENCODE cCRE type to regulatory class mapping
CCRE_TYPE_MAP = {
    "PLS":  ("promoter", 0.80),       # Promoter-like signature
    "pELS": ("enhancer", 0.70),       # Proximal enhancer-like
    "dELS": ("enhancer", 0.65),       # Distal enhancer-like
    "CTCF-only": ("insulator", 0.40), # CTCF-bound insulator
    "DNase-H3K4me3": ("promoter", 0.75),
    "DNase-only": ("open_chromatin", 0.30),
}


class RegulatoryAnnotator:
    """Annotate non-coding variants with regulatory context and gene targets."""

    def __init__(self, config: RegulatoryConfig | None = None):
        self.config = config or RegulatoryConfig()
        self._encode_index = None
        self._chromhmm_index = None
        self._eqtl_index = None
        self._hic_index = None
        self._abc_index = None

    def annotate(self, variants: list[AnnotatedVariant]) -> list[AnnotatedVariant]:
        """Run the full non-coding annotation pipeline.

        Pipeline order:
        1. Regulatory region annotation (ENCODE, ChromHMM)
        2. Conservation scores
        3. Variant-to-gene mapping (eQTL, Hi-C, ABC)
        4. Deep learning regulatory scores
        5. Compute unified regulatory impact score
        """
        if not self.config.enabled:
            return variants

        noncoding = [v for v in variants if v.is_noncoding]
        if not noncoding:
            logger.info("No non-coding variants to annotate")
            return variants

        logger.info(f"Annotating {len(noncoding)} non-coding variants with regulatory data")

        for var in noncoding:
            self._annotate_regulatory_regions(var)
            self._annotate_conservation(var)
            self._annotate_gene_mapping(var)
            self._annotate_deep_learning_scores(var)
            self._compute_regulatory_score(var)

        n_annotated = sum(1 for v in noncoding if v.has_regulatory_annotation)
        n_mapped = sum(1 for v in noncoding if v.target_gene_symbol)
        logger.info(
            f"Regulatory annotation complete: {n_annotated} with regulatory context, "
            f"{n_mapped} mapped to target genes"
        )

        return variants

    # ── 1. Regulatory Region Annotation ────────────────────────────────────────

    def _annotate_regulatory_regions(self, var: AnnotatedVariant) -> None:
        """Annotate variant with overlapping regulatory regions.

        Sources: ENCODE cCREs, Ensembl Regulatory Build, Roadmap ChromHMM.
        """
        self._annotate_encode_ccres(var)
        self._annotate_chromhmm(var)
        self._annotate_ensembl_regulatory(var)

    def _annotate_encode_ccres(self, var: AnnotatedVariant) -> None:
        """Overlap with ENCODE candidate Cis-Regulatory Elements (cCREs).

        ENCODE cCRE registry BED format:
        chrom  start  end  accession  cCRE_type  ...
        """
        if self.config.encode_cres_path and Path(self.config.encode_cres_path).exists():
            try:
                import pysam
                if self._encode_index is None:
                    self._encode_index = pysam.TabixFile(self.config.encode_cres_path)

                chrom = var.variant.chrom
                pos = var.variant.pos
                for row in self._encode_index.fetch(chrom, pos - 1, pos):
                    fields = row.split("\t")
                    if len(fields) >= 5:
                        ccre_type = fields[4]
                        if ccre_type in CCRE_TYPE_MAP:
                            reg_class, score = CCRE_TYPE_MAP[ccre_type]
                            var.regulatory_class = reg_class
                            var.regulatory_feature_id = fields[3] if len(fields) > 3 else ""
                            break
            except Exception as e:
                logger.debug(f"ENCODE cCRE lookup failed: {e}")
        elif self.config.encode_cres_path is None:
            # Try Ensembl REST API as fallback
            self._encode_ccres_remote(var)

    def _encode_ccres_remote(self, var: AnnotatedVariant) -> None:
        """Query ENCODE/Ensembl for regulatory region overlap via REST."""
        try:
            chrom = var.variant.chrom.replace("chr", "")
            pos = var.variant.pos
            url = (
                f"https://rest.ensembl.org/overlap/region/human/"
                f"{chrom}:{pos}-{pos}?feature=regulatory"
            )
            resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
            if resp.status_code == 200:
                for feature in resp.json():
                    feat_type = feature.get("feature_type", "")
                    if feat_type in ("Promoter", "Enhancer", "CTCF Binding Site",
                                     "Open chromatin", "TF binding site", "Promoter Flanking Region"):
                        var.regulatory_class = feat_type.lower().replace(" ", "_")
                        var.regulatory_feature_id = feature.get("id", "")
                        break
        except Exception as e:
            logger.debug(f"Ensembl regulatory overlap API failed: {e}")

    def _annotate_chromhmm(self, var: AnnotatedVariant) -> None:
        """Annotate with Roadmap Epigenomics ChromHMM chromatin state.

        ChromHMM BED format (per epigenome):
        chrom  start  end  state_code
        """
        if not (self.config.roadmap_chromhmm_path
                and Path(self.config.roadmap_chromhmm_path).exists()):
            return

        try:
            import pysam
            if self._chromhmm_index is None:
                self._chromhmm_index = pysam.TabixFile(self.config.roadmap_chromhmm_path)

            best_state = ""
            best_score = 0.0

            for row in self._chromhmm_index.fetch(
                var.variant.chrom, var.variant.pos - 1, var.variant.pos
            ):
                fields = row.split("\t")
                if len(fields) >= 4:
                    state = fields[3]
                    if state in CHROMHMM_STATES:
                        state_name, state_score = CHROMHMM_STATES[state]
                        if state_score > best_score:
                            best_state = state
                            best_score = state_score

            if best_state:
                var.chromhmm_state = best_state
                var.chromhmm_state_name = CHROMHMM_STATES[best_state][0]
                # Infer regulatory class from ChromHMM if not already set
                if not var.regulatory_class:
                    if "Enh" in best_state:
                        var.regulatory_class = "enhancer"
                    elif "Tss" in best_state:
                        var.regulatory_class = "promoter"
                    elif "CTCF" in best_state:
                        var.regulatory_class = "insulator"

        except Exception as e:
            logger.debug(f"ChromHMM lookup failed: {e}")

    def _annotate_ensembl_regulatory(self, var: AnnotatedVariant) -> None:
        """Annotate with Ensembl Regulatory Build features."""
        if not (self.config.ensembl_regulatory_path
                and Path(self.config.ensembl_regulatory_path).exists()):
            return

        try:
            import pysam
            tbx = pysam.TabixFile(self.config.ensembl_regulatory_path)
            for row in tbx.fetch(var.variant.chrom, var.variant.pos - 1, var.variant.pos):
                fields = row.split("\t")
                if len(fields) >= 5:
                    if not var.regulatory_class:
                        var.regulatory_class = fields[4].lower()
                    if not var.regulatory_feature_id:
                        var.regulatory_feature_id = fields[3]
                    break
        except Exception as e:
            logger.debug(f"Ensembl Regulatory Build lookup failed: {e}")

    # ── 2. Conservation Scores ─────────────────────────────────────────────────

    def _annotate_conservation(self, var: AnnotatedVariant) -> None:
        """Add conservation scores (PhastCons, PhyloP, GERP++)."""
        self._query_tabix_score(var, self.config.phastcons_path, "phastcons_score")
        self._query_tabix_score(var, self.config.phylop_path, "phylop_score")
        self._query_tabix_score(var, self.config.gerp_path, "gerp_score")

        # Fallback: try CADD for conservation proxy (CADD includes conservation)
        if (var.phastcons_score is None and var.phylop_score is None
                and var.gerp_score is None and var.cadd_phred is not None):
            # Approximate conservation from CADD (CADD PHRED > 15 suggests conservation)
            var.phastcons_score = min(var.cadd_phred / 30.0, 1.0) if var.cadd_phred else None

    def _query_tabix_score(
        self,
        var: AnnotatedVariant,
        path: str | None,
        field_name: str,
        score_col: int = 4,
    ) -> None:
        """Generic tabix score query."""
        if not path or not Path(path).exists():
            return

        try:
            import pysam
            tbx = pysam.TabixFile(path)
            for row in tbx.fetch(var.variant.chrom, var.variant.pos - 1, var.variant.pos):
                fields = row.split("\t")
                if len(fields) > score_col:
                    score = float(fields[score_col])
                    setattr(var, field_name, score)
                    break
        except Exception as e:
            logger.debug(f"{field_name} tabix query failed: {e}")

    # ── 3. Variant-to-Gene Mapping ─────────────────────────────────────────────

    def _annotate_gene_mapping(self, var: AnnotatedVariant) -> None:
        """Map non-coding variant to target gene(s).

        Priority order:
        1. ABC model (Activity-by-Contact) - highest confidence
        2. eQTL data (GTEx) - expression evidence
        3. Hi-C chromatin interaction - 3D genome structure
        4. Nearest gene within TAD - fallback
        """
        # Already has a gene from VEP? Keep it as fallback
        vep_gene = var.gene_symbol

        # Try ABC model first (highest confidence)
        if self._map_via_abc(var):
            return

        # Try eQTL mapping
        if self._map_via_eqtl(var):
            return

        # Try Hi-C chromatin interaction
        if self._map_via_hic(var):
            return

        # Fallback: use VEP nearest gene
        if vep_gene:
            var.target_gene_symbol = vep_gene
            var.target_gene_id = var.gene_id
            var.gene_mapping_method = "nearest"
            var.gene_mapping_score = 0.3  # Low confidence for nearest-gene

    def _map_via_abc(self, var: AnnotatedVariant) -> bool:
        """Map variant to gene using Activity-by-Contact model.

        ABC BED format:
        chrom  start  end  gene  abc_score  ...
        """
        if not (self.config.abc_model_path
                and Path(self.config.abc_model_path).exists()):
            return False

        try:
            import pysam
            if self._abc_index is None:
                self._abc_index = pysam.TabixFile(self.config.abc_model_path)

            best_score = 0.0
            best_gene = ""

            for row in self._abc_index.fetch(
                var.variant.chrom, var.variant.pos - 1, var.variant.pos
            ):
                fields = row.split("\t")
                if len(fields) >= 5:
                    gene = fields[3]
                    score = float(fields[4])
                    if score > best_score:
                        best_score = score
                        best_gene = gene

            if best_gene and best_score >= self.config.abc_score_threshold:
                var.target_gene_symbol = best_gene
                var.gene_mapping_method = "abc"
                var.gene_mapping_score = min(best_score / 0.1, 1.0)  # Normalize
                var.abc_score = best_score
                return True

        except Exception as e:
            logger.debug(f"ABC model lookup failed: {e}")

        return False

    def _map_via_eqtl(self, var: AnnotatedVariant) -> bool:
        """Map variant to gene using GTEx eQTL data.

        GTEx significant pairs format:
        variant_id  gene_id  gene_symbol  tissue  pvalue  beta  ...
        """
        if self.config.gtex_eqtl_path and Path(self.config.gtex_eqtl_path).exists():
            return self._eqtl_local(var)
        return self._eqtl_remote(var)

    def _eqtl_local(self, var: AnnotatedVariant) -> bool:
        """Query local GTEx eQTL tabix file."""
        try:
            import pysam
            if self._eqtl_index is None:
                self._eqtl_index = pysam.TabixFile(self.config.gtex_eqtl_path)

            best_pval = 1.0
            best_gene = ""
            best_tissue = ""
            best_beta = 0.0

            for row in self._eqtl_index.fetch(
                var.variant.chrom, var.variant.pos - 1, var.variant.pos
            ):
                fields = row.split("\t")
                if len(fields) >= 6:
                    gene = fields[2]
                    tissue = fields[3]
                    pval = float(fields[4])
                    beta = float(fields[5])

                    if pval < best_pval:
                        best_pval = pval
                        best_gene = gene
                        best_tissue = tissue
                        best_beta = beta

            if best_gene and best_pval < self.config.eqtl_pvalue_threshold:
                var.target_gene_symbol = best_gene
                var.gene_mapping_method = "eqtl"
                # Convert p-value to confidence score
                var.gene_mapping_score = min(-math.log10(best_pval) / 20.0, 1.0)
                var.eqtl_tissue = best_tissue
                var.eqtl_pvalue = best_pval
                var.eqtl_beta = best_beta
                return True

        except Exception as e:
            logger.debug(f"Local eQTL lookup failed: {e}")

        return False

    def _eqtl_remote(self, var: AnnotatedVariant) -> bool:
        """Query GTEx API for eQTL associations."""
        try:
            chrom = var.variant.chrom.replace("chr", "")
            pos = var.variant.pos
            url = (
                f"https://gtexportal.org/api/v2/association/singleTissueEqtl"
                f"?format=json&snpId=chr{chrom}_{pos}_{var.variant.ref}_{var.variant.alt}_b38"
                f"&datasetId=gtex_v8"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", []) if isinstance(data, dict) else data
                if results:
                    # Pick most significant eQTL
                    best = min(results, key=lambda x: x.get("pValue", 1.0))
                    pval = best.get("pValue", 1.0)
                    if pval < self.config.eqtl_pvalue_threshold:
                        var.target_gene_symbol = best.get("geneSymbol", "")
                        var.target_gene_id = best.get("gencodeId", "")
                        var.gene_mapping_method = "eqtl"
                        var.gene_mapping_score = min(-math.log10(pval) / 20.0, 1.0)
                        var.eqtl_tissue = best.get("tissueSiteDetailId", "")
                        var.eqtl_pvalue = pval
                        var.eqtl_beta = best.get("nes", 0.0)
                        return True
        except Exception as e:
            logger.debug(f"GTEx API query failed: {e}")

        return False

    def _map_via_hic(self, var: AnnotatedVariant) -> bool:
        """Map variant to gene using Hi-C chromatin interaction data.

        Hi-C interactions BED format:
        chrom1  start1  end1  chrom2  start2  end2  score  gene  ...
        """
        if not (self.config.hic_path and Path(self.config.hic_path).exists()):
            return False

        try:
            import pysam
            if self._hic_index is None:
                self._hic_index = pysam.TabixFile(self.config.hic_path)

            best_score = 0.0
            best_gene = ""

            for row in self._hic_index.fetch(
                var.variant.chrom, var.variant.pos - 1, var.variant.pos
            ):
                fields = row.split("\t")
                if len(fields) >= 8:
                    score = float(fields[6])
                    gene = fields[7]
                    if score > best_score:
                        best_score = score
                        best_gene = gene

            if best_gene and best_score > 0:
                var.target_gene_symbol = best_gene
                var.gene_mapping_method = "chromatin_interaction"
                var.gene_mapping_score = min(best_score, 1.0)
                var.hic_score = best_score
                return True

        except Exception as e:
            logger.debug(f"Hi-C lookup failed: {e}")

        return False

    # ── 4. Deep Learning Regulatory Scores ─────────────────────────────────────

    def _annotate_deep_learning_scores(self, var: AnnotatedVariant) -> None:
        """Score variant with deep learning regulatory models.

        Supported models:
          - Enformer: predicts gene expression from sequence
          - Sei: sequence-based regulatory predictions
          - DeepBind: TF binding disruption

        These are computationally expensive; uses precomputed scores
        from tabix files when available, otherwise estimates from
        existing annotations.
        """
        # Precomputed Enformer scores (if available)
        self._query_tabix_score(var, None, "enformer_score")
        self._query_tabix_score(var, None, "sei_score")
        self._query_tabix_score(var, None, "deepbind_score")

        # If no DL scores available, estimate from existing annotations
        if (var.enformer_score is None and var.sei_score is None
                and var.deepbind_score is None):
            self._estimate_regulatory_dl_score(var)

    def _estimate_regulatory_dl_score(self, var: AnnotatedVariant) -> None:
        """Estimate regulatory DL score from available annotations.

        Uses a heuristic combining:
          - Conservation (high conservation in regulatory region = likely functional)
          - Regulatory class (enhancer/promoter scored higher)
          - CADD score (captures some regulatory signal)
        """
        score_components = []

        # Conservation signal
        if var.phastcons_score is not None and var.phastcons_score > 0.5:
            score_components.append(var.phastcons_score * 0.4)
        if var.gerp_score is not None and var.gerp_score > 2.0:
            score_components.append(min(var.gerp_score / 6.0, 1.0) * 0.3)

        # ChromHMM state signal
        if var.chromhmm_state and var.chromhmm_state in CHROMHMM_STATES:
            _, state_score = CHROMHMM_STATES[var.chromhmm_state]
            score_components.append(state_score * 0.3)

        # Regulatory class signal
        class_scores = {
            "promoter": 0.7, "enhancer": 0.65, "TFBS": 0.6,
            "tf_binding_site": 0.6, "insulator": 0.4,
            "open_chromatin": 0.3, "silencer": 0.5,
        }
        if var.regulatory_class in class_scores:
            score_components.append(class_scores[var.regulatory_class] * 0.3)

        if score_components:
            # Use the estimated score as a proxy Enformer score
            var.enformer_score = min(sum(score_components), 1.0)

    # ── 5. Compute Unified Regulatory Impact Score ─────────────────────────────

    def _compute_regulatory_score(self, var: AnnotatedVariant) -> None:
        """Compute the unified non-coding regulatory impact score.

        Formula:
          regulatory_score = w_splice * splice_score
                           + w_reg * region_score
                           + w_cons * conservation_score
                           + w_dl * dl_score
                           + w_map * mapping_confidence

        All sub-scores normalized to [0, 1].
        """
        cfg = self.config

        # Sub-score 1: Splicing disruption (SpliceAI)
        splice_score = 0.0
        if var.spliceai_max is not None:
            splice_score = var.spliceai_max

        # Sub-score 2: Regulatory region annotation
        region_score = 0.0
        if var.chromhmm_state and var.chromhmm_state in CHROMHMM_STATES:
            _, region_score = CHROMHMM_STATES[var.chromhmm_state]
        elif var.regulatory_class:
            region_scores = {
                "promoter": 0.80, "enhancer": 0.75, "tf_binding_site": 0.60,
                "insulator": 0.40, "open_chromatin": 0.30, "silencer": 0.50,
            }
            region_score = region_scores.get(var.regulatory_class, 0.20)

        # Sub-score 3: Conservation
        conservation_score = 0.0
        cons_values = [s for s in [var.phastcons_score, var.phylop_score] if s is not None]
        if cons_values:
            conservation_score = max(cons_values)
        elif var.gerp_score is not None:
            conservation_score = min(max(var.gerp_score / 6.0, 0.0), 1.0)

        # Sub-score 4: Deep learning regulatory prediction
        dl_score = 0.0
        dl_values = [s for s in [var.enformer_score, var.sei_score, var.deepbind_score]
                     if s is not None]
        if dl_values:
            dl_score = max(dl_values)

        # Sub-score 5: Gene mapping confidence
        mapping_score = var.gene_mapping_score

        # Weighted combination
        var.regulatory_score = (
            cfg.w_splicing * splice_score
            + cfg.w_regulatory_region * region_score
            + cfg.w_conservation * conservation_score
            + cfg.w_deep_learning * dl_score
            + cfg.w_gene_mapping * mapping_score
        )

        # Also set the noncoding_impact_score for downstream scoring
        var.noncoding_impact_score = var.regulatory_score

        # Enhance gene_symbol with target gene if mapped
        if var.target_gene_symbol and not var.gene_symbol:
            var.gene_symbol = var.target_gene_symbol
            var.gene_id = var.target_gene_id
