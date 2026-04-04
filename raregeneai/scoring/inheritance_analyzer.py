"""Trio-based inheritance model analysis.

Classifies variants by inheritance pattern using parental genotypes:
  - De novo: absent in both parents (highest weight)
  - Compound heterozygous: two het variants in trans (biallelic)
  - Homozygous recessive: HOM_ALT in proband, HET in both parents
  - X-linked: hemizygous in male proband, carrier mother
  - Inherited dominant: present in an affected parent

Tags each AnnotatedVariant in-place with its inheritance class
and computes per-gene inheritance scores for the ranking engine.
"""

from __future__ import annotations

from collections import defaultdict

from loguru import logger

from raregeneai.models.data_models import (
    AnnotatedVariant,
    FunctionalImpact,
    InheritanceMode,
    Pedigree,
    Zygosity,
)


# ── Inheritance score table ───────────────────────────────────────────────────
# These scores reflect clinical actionability of each inheritance class.
# De novo LoF is the strongest signal in rare disease genetics.
INHERITANCE_WEIGHTS = {
    "de_novo_lof":          1.00,  # De novo + loss-of-function
    "de_novo_missense":     0.90,  # De novo + missense (moderate)
    "de_novo_other":        0.80,  # De novo + other consequence
    "compound_het_lof_lof": 0.95,  # Compound het: both LoF
    "compound_het_lof_mis": 0.85,  # Compound het: LoF + missense
    "compound_het_mis_mis": 0.75,  # Compound het: both missense
    "compound_het_other":   0.65,  # Compound het: other combinations
    "hom_recessive_lof":    0.90,  # Homozygous recessive LoF
    "hom_recessive_other":  0.75,  # Homozygous recessive missense/other
    "x_linked_hemi_lof":    0.95,  # X-linked hemizygous LoF (male)
    "x_linked_hemi_other":  0.80,  # X-linked hemizygous other
    "inherited_dominant":   0.40,  # Inherited from affected parent
    "unknown_het":          0.30,  # Heterozygous, no trio data
    "unknown_hom":          0.50,  # Homozygous, no trio data
    "unknown_other":        0.20,  # Other/unknown
}


class InheritanceAnalyzer:
    """Analyze variant inheritance patterns in family context."""

    def analyze_trio(
        self,
        proband_variants: list[AnnotatedVariant],
        father_variants: list[AnnotatedVariant] | None = None,
        mother_variants: list[AnnotatedVariant] | None = None,
        pedigree: Pedigree | None = None,
    ) -> dict[str, list[dict]]:
        """Analyze inheritance in a trio and tag variants in-place.

        Each AnnotatedVariant in proband_variants gets its inheritance
        fields populated: inheritance_class, is_de_novo, is_compound_het,
        is_hom_recessive, parent_of_origin, inheritance_score.

        Returns categorized variants dict (for backward compatibility).
        """
        results: dict[str, list[dict]] = {
            "de_novo": [],
            "compound_het": [],
            "homozygous_recessive": [],
            "x_linked": [],
            "dominant": [],
        }

        has_parents = father_variants is not None or mother_variants is not None

        if not has_parents:
            logger.info("No parental data; using zygosity-based inheritance scoring")
            # Tag with zygosity-only scores
            for var in proband_variants:
                self._tag_no_trio(var)
            return results

        logger.info("Running trio-based inheritance analysis")

        # Build lookup for parental variants
        father_keys = self._build_variant_lookup(father_variants or [])
        mother_keys = self._build_variant_lookup(mother_variants or [])

        # Group proband variants by gene (using effective_gene_symbol)
        gene_variants: dict[str, list[AnnotatedVariant]] = defaultdict(list)
        for var in proband_variants:
            gene = var.effective_gene_symbol
            if gene:
                gene_variants[gene].append(var)

        # ── Classify each variant ─────────────────────────────────────────
        for var in proband_variants:
            key = var.variant.variant_key
            in_father = key in father_keys
            in_mother = key in mother_keys

            entry = {
                "variant": var,
                "gene": var.effective_gene_symbol,
                "in_father": in_father,
                "in_mother": in_mother,
            }

            # De novo: absent in both parents
            if not in_father and not in_mother:
                var.inheritance_class = "de_novo"
                var.is_de_novo = True
                var.is_inherited = False
                var.parent_of_origin = ""
                results["de_novo"].append(entry)

            # Homozygous recessive: HOM_ALT in proband, HET in both parents
            elif (
                var.variant.zygosity == Zygosity.HOM_ALT
                and in_father
                and in_mother
            ):
                father_zyg = father_keys[key].variant.zygosity
                mother_zyg = mother_keys[key].variant.zygosity
                if father_zyg == Zygosity.HET and mother_zyg == Zygosity.HET:
                    var.inheritance_class = "hom_recessive"
                    var.is_hom_recessive = True
                    var.is_inherited = True
                    var.parent_of_origin = "both"
                    results["homozygous_recessive"].append(entry)
                else:
                    var.inheritance_class = "inherited_dominant"
                    var.is_inherited = True
                    var.parent_of_origin = "both"

            # X-linked
            elif var.variant.chrom in ("X", "chrX"):
                var.is_inherited = in_father or in_mother
                if in_mother and not in_father:
                    var.parent_of_origin = "maternal"
                elif in_father and not in_mother:
                    var.parent_of_origin = "paternal"
                var.inheritance_class = "x_linked"
                results["x_linked"].append(entry)

            # Inherited heterozygous
            elif var.variant.zygosity == Zygosity.HET:
                var.inheritance_class = "inherited_dominant"
                var.is_inherited = True
                if in_father and not in_mother:
                    var.parent_of_origin = "paternal"
                elif in_mother and not in_father:
                    var.parent_of_origin = "maternal"
                else:
                    var.parent_of_origin = "both"
                results["dominant"].append(entry)

            else:
                var.inheritance_class = "inherited_dominant"
                var.is_inherited = True

        # ── Compound heterozygous analysis ────────────────────────────────
        for gene, vars_in_gene in gene_variants.items():
            het_vars = [v for v in vars_in_gene if v.variant.zygosity == Zygosity.HET]
            if len(het_vars) >= 2:
                comp_het_pairs = self._find_compound_hets(
                    het_vars, father_keys, mother_keys
                )
                for v1, v2 in comp_het_pairs:
                    # Tag both variants
                    v1.is_compound_het = True
                    v1.inheritance_class = "compound_het"
                    v1.compound_het_partner_key = v2.variant.variant_key

                    v2.is_compound_het = True
                    v2.inheritance_class = "compound_het"
                    v2.compound_het_partner_key = v1.variant.variant_key

                    results["compound_het"].append({
                        "gene": gene,
                        "variant_1": v1,
                        "variant_2": v2,
                    })

        # ── Assign inheritance scores to all variants ─────────────────────
        for var in proband_variants:
            var.inheritance_score = self._compute_variant_inheritance_score(var)

        # Summary
        for category, items in results.items():
            if items:
                logger.info(f"Inheritance - {category}: {len(items)} variants/pairs")

        return results

    def _tag_no_trio(self, var: AnnotatedVariant) -> None:
        """Tag variant with zygosity-only inheritance score (no trio data)."""
        zyg = var.variant.zygosity
        if zyg == Zygosity.HOM_ALT:
            var.inheritance_class = "unknown_hom"
            var.inheritance_score = INHERITANCE_WEIGHTS["unknown_hom"]
        elif zyg == Zygosity.HET:
            var.inheritance_class = "unknown_het"
            var.inheritance_score = INHERITANCE_WEIGHTS["unknown_het"]
        elif zyg == Zygosity.HEMIZYGOUS:
            var.inheritance_class = "x_linked"
            var.inheritance_score = INHERITANCE_WEIGHTS["x_linked_hemi_other"]
        else:
            var.inheritance_class = "unknown_other"
            var.inheritance_score = INHERITANCE_WEIGHTS["unknown_other"]

    def _compute_variant_inheritance_score(self, var: AnnotatedVariant) -> float:
        """Compute inheritance score for a single variant based on its class + impact."""
        is_lof = var.impact == FunctionalImpact.HIGH
        is_missense = var.impact == FunctionalImpact.MODERATE

        if var.is_de_novo:
            if is_lof:
                return INHERITANCE_WEIGHTS["de_novo_lof"]
            elif is_missense:
                return INHERITANCE_WEIGHTS["de_novo_missense"]
            return INHERITANCE_WEIGHTS["de_novo_other"]

        if var.is_compound_het:
            # Score based on the impact of this specific allele
            # (the pair-level score is computed at gene level)
            if is_lof:
                return INHERITANCE_WEIGHTS["compound_het_lof_lof"]
            elif is_missense:
                return INHERITANCE_WEIGHTS["compound_het_mis_mis"]
            return INHERITANCE_WEIGHTS["compound_het_other"]

        if var.is_hom_recessive:
            if is_lof:
                return INHERITANCE_WEIGHTS["hom_recessive_lof"]
            return INHERITANCE_WEIGHTS["hom_recessive_other"]

        if var.inheritance_class == "x_linked":
            if var.variant.zygosity == Zygosity.HEMIZYGOUS:
                if is_lof:
                    return INHERITANCE_WEIGHTS["x_linked_hemi_lof"]
                return INHERITANCE_WEIGHTS["x_linked_hemi_other"]

        if var.inheritance_class == "inherited_dominant":
            return INHERITANCE_WEIGHTS["inherited_dominant"]

        # No trio data fallback
        return self._zygosity_only_score(var)

    def _zygosity_only_score(self, var: AnnotatedVariant) -> float:
        """Fallback: score based on zygosity alone (no trio)."""
        zyg = var.variant.zygosity
        if zyg == Zygosity.HOM_ALT:
            return INHERITANCE_WEIGHTS["unknown_hom"]
        elif zyg == Zygosity.HET:
            return INHERITANCE_WEIGHTS["unknown_het"]
        elif zyg == Zygosity.HEMIZYGOUS:
            return INHERITANCE_WEIGHTS["x_linked_hemi_other"]
        return INHERITANCE_WEIGHTS["unknown_other"]

    def compute_gene_inheritance_score(
        self,
        gene_symbol: str,
        variants: list[AnnotatedVariant],
    ) -> dict:
        """Compute gene-level inheritance evidence summary.

        Returns dict with:
          - inheritance_score: max variant inheritance score
          - has_de_novo: bool
          - has_de_novo_lof: bool
          - has_compound_het: bool
          - has_hom_recessive: bool
          - trio_analyzed: bool (whether trio data was available)
          - inheritance_class: best inheritance class for this gene
        """
        if not variants:
            return {"inheritance_score": 0.0, "trio_analyzed": False}

        trio_analyzed = any(
            v.inheritance_class not in ("", "unknown_het", "unknown_hom", "unknown_other")
            for v in variants
        )

        has_de_novo = any(v.is_de_novo for v in variants)
        has_de_novo_lof = any(
            v.is_de_novo and v.impact == FunctionalImpact.HIGH for v in variants
        )
        has_compound_het = any(v.is_compound_het for v in variants)
        has_hom_recessive = any(v.is_hom_recessive for v in variants)

        # Best (highest) inheritance score across variants
        max_score = max((v.inheritance_score for v in variants), default=0.0)

        # For compound hets: compute pair-level score
        if has_compound_het:
            pair_score = self._compound_het_pair_score(gene_symbol, variants)
            max_score = max(max_score, pair_score)

        # Determine the dominant inheritance class for the gene
        if has_de_novo_lof:
            best_class = "de_novo_lof"
        elif has_de_novo:
            best_class = "de_novo"
        elif has_compound_het:
            best_class = "compound_het"
        elif has_hom_recessive:
            best_class = "hom_recessive"
        else:
            best_class = max(
                variants,
                key=lambda v: v.inheritance_score,
            ).inheritance_class

        return {
            "inheritance_score": max_score,
            "has_de_novo": has_de_novo,
            "has_de_novo_lof": has_de_novo_lof,
            "has_compound_het": has_compound_het,
            "has_hom_recessive": has_hom_recessive,
            "trio_analyzed": trio_analyzed,
            "inheritance_class": best_class,
        }

    def _compound_het_pair_score(
        self, gene_symbol: str, variants: list[AnnotatedVariant],
    ) -> float:
        """Compute the compound-het pair score based on combined impact."""
        comp_het_vars = [v for v in variants if v.is_compound_het]
        if len(comp_het_vars) < 2:
            return INHERITANCE_WEIGHTS["compound_het_other"]

        impacts = sorted(
            [v.impact for v in comp_het_vars],
            key=lambda i: {"HIGH": 3, "MODERATE": 2, "LOW": 1, "MODIFIER": 0}.get(i.value, 0),
            reverse=True,
        )

        top_two = impacts[:2]
        if top_two[0] == FunctionalImpact.HIGH and top_two[1] == FunctionalImpact.HIGH:
            return INHERITANCE_WEIGHTS["compound_het_lof_lof"]
        elif top_two[0] == FunctionalImpact.HIGH:
            return INHERITANCE_WEIGHTS["compound_het_lof_mis"]
        elif top_two[0] == FunctionalImpact.MODERATE and top_two[1] == FunctionalImpact.MODERATE:
            return INHERITANCE_WEIGHTS["compound_het_mis_mis"]
        return INHERITANCE_WEIGHTS["compound_het_other"]

    def _find_compound_hets(
        self,
        het_variants: list[AnnotatedVariant],
        father_keys: dict[str, AnnotatedVariant],
        mother_keys: dict[str, AnnotatedVariant],
    ) -> list[tuple[AnnotatedVariant, AnnotatedVariant]]:
        """Find compound heterozygous pairs (variants in trans)."""
        pairs = []

        for i, v1 in enumerate(het_variants):
            for v2 in het_variants[i + 1:]:
                k1 = v1.variant.variant_key
                k2 = v2.variant.variant_key

                # In trans = one from father, one from mother
                v1_paternal = k1 in father_keys and k1 not in mother_keys
                v1_maternal = k1 in mother_keys and k1 not in father_keys
                v2_paternal = k2 in father_keys and k2 not in mother_keys
                v2_maternal = k2 in mother_keys and k2 not in father_keys

                if (v1_paternal and v2_maternal) or (v1_maternal and v2_paternal):
                    pairs.append((v1, v2))

        return pairs

    @staticmethod
    def _build_variant_lookup(
        variants: list[AnnotatedVariant],
    ) -> dict[str, AnnotatedVariant]:
        return {v.variant.variant_key: v for v in variants}

    def infer_inheritance_modes(
        self,
        gene_symbol: str,
        variants: list[AnnotatedVariant],
    ) -> list[InheritanceMode]:
        """Infer compatible inheritance modes using trio tags when available."""
        modes = set()

        for var in variants:
            chrom = var.variant.chrom
            zyg = var.variant.zygosity

            # Use trio-based class if available
            if var.is_de_novo:
                modes.add(InheritanceMode.AUTOSOMAL_DOMINANT)
            elif var.is_hom_recessive:
                modes.add(InheritanceMode.AUTOSOMAL_RECESSIVE)
            elif var.is_compound_het:
                modes.add(InheritanceMode.AUTOSOMAL_RECESSIVE)
            elif chrom in ("X", "chrX"):
                if zyg == Zygosity.HEMIZYGOUS:
                    modes.add(InheritanceMode.X_LINKED_RECESSIVE)
                elif zyg == Zygosity.HET:
                    modes.add(InheritanceMode.X_LINKED_DOMINANT)
            elif chrom in ("MT", "chrM"):
                modes.add(InheritanceMode.MITOCHONDRIAL)
            else:
                if zyg == Zygosity.HOM_ALT:
                    modes.add(InheritanceMode.AUTOSOMAL_RECESSIVE)
                elif zyg == Zygosity.HET:
                    modes.add(InheritanceMode.AUTOSOMAL_DOMINANT)
                    het_count = sum(
                        1 for v in variants
                        if v.variant.zygosity == Zygosity.HET
                        and v.effective_gene_symbol == gene_symbol
                    )
                    if het_count >= 2:
                        modes.add(InheritanceMode.AUTOSOMAL_RECESSIVE)

        return list(modes)
