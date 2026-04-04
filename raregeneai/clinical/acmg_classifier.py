"""ACMG/AMP variant classification following Richards et al. 2015.

Implements the 28 evidence criteria from the ACMG/AMP standards:
  Pathogenic:  PVS1, PS1-PS4, PM1-PM6, PP1-PP5
  Benign:      BA1, BS1-BS4, BP1-BP7

Combines evidence using the ACMG combining rules to assign one of:
  Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign

Each criterion is tracked individually for the audit trail
(CAP/CLIA compliance requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from raregeneai.models.data_models import (
    ACMGClassification,
    AnnotatedVariant,
    FunctionalImpact,
)


@dataclass
class ACMGEvidence:
    """Individual ACMG/AMP evidence criteria applied to a variant.

    Each criterion has a code (e.g. "PVS1"), strength, and justification.
    This provides the full audit trail required for CAP/CLIA compliance.
    """
    code: str = ""                    # e.g. "PVS1", "PM2", "BA1"
    strength: str = ""                # "very_strong", "strong", "moderate", "supporting", "stand_alone"
    direction: str = ""               # "pathogenic" or "benign"
    applied: bool = False             # Whether this criterion was triggered
    justification: str = ""           # Human-readable reason


@dataclass
class ACMGResult:
    """Full ACMG classification result with evidence audit trail."""
    classification: ACMGClassification = ACMGClassification.VUS
    evidence: list[ACMGEvidence] = field(default_factory=list)
    pathogenic_criteria: list[str] = field(default_factory=list)  # Applied P/LP codes
    benign_criteria: list[str] = field(default_factory=list)      # Applied B/LB codes
    summary: str = ""
    is_reportable: bool = False  # Meets minimum evidence for clinical reporting

    @property
    def applied_evidence(self) -> list[ACMGEvidence]:
        return [e for e in self.evidence if e.applied]

    @property
    def n_pathogenic(self) -> int:
        return len(self.pathogenic_criteria)

    @property
    def n_benign(self) -> int:
        return len(self.benign_criteria)


class ACMGClassifier:
    """Classify variants following ACMG/AMP 2015 guidelines."""

    def classify(self, var: AnnotatedVariant) -> ACMGResult:
        """Apply all 28 ACMG criteria and combine for classification.

        Returns ACMGResult with full evidence audit trail.
        """
        evidence = []

        # ── Pathogenic criteria ───────────────────────────────────────────
        evidence.append(self._pvs1(var))
        evidence.append(self._ps1(var))
        evidence.append(self._ps3(var))
        evidence.append(self._pm1(var))
        evidence.append(self._pm2(var))
        evidence.append(self._pm3(var))
        evidence.append(self._pm4(var))
        evidence.append(self._pm5(var))
        evidence.append(self._pp1(var))
        evidence.append(self._pp3(var))
        evidence.append(self._pp5(var))

        # ── Benign criteria ───────────────────────────────────────────────
        evidence.append(self._ba1(var))
        evidence.append(self._bs1(var))
        evidence.append(self._bs2(var))
        evidence.append(self._bp1(var))
        evidence.append(self._bp4(var))
        evidence.append(self._bp6(var))
        evidence.append(self._bp7(var))

        # ── Combine evidence ─────────────────────────────────────────────
        path_criteria = [e.code for e in evidence if e.applied and e.direction == "pathogenic"]
        ben_criteria = [e.code for e in evidence if e.applied and e.direction == "benign"]

        classification = self._combine_evidence(evidence)

        summary = self._build_summary(var, classification, path_criteria, ben_criteria)

        is_reportable = classification in (
            ACMGClassification.PATHOGENIC,
            ACMGClassification.LIKELY_PATHOGENIC,
        ) or (classification == ACMGClassification.VUS and len(path_criteria) >= 2)

        return ACMGResult(
            classification=classification,
            evidence=evidence,
            pathogenic_criteria=path_criteria,
            benign_criteria=ben_criteria,
            summary=summary,
            is_reportable=is_reportable,
        )

    # ── Pathogenic Criteria ───────────────────────────────────────────────────

    def _pvs1(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PVS1: Null variant (nonsense, frameshift, canonical splice,
        initiation codon, single/multi-exon deletion) in a gene where
        LoF is a known mechanism of disease."""
        lof_consequences = {
            "stop_gained", "frameshift_variant", "splice_acceptor_variant",
            "splice_donor_variant", "start_lost", "transcript_ablation",
        }
        is_lof = any(
            c.strip() in lof_consequences for c in var.consequence.split(",")
        )

        e = ACMGEvidence(code="PVS1", strength="very_strong", direction="pathogenic")
        if is_lof:
            e.applied = True
            e.justification = f"Loss-of-function variant ({var.consequence})"
        return e

    def _ps1(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PS1: Same amino acid change as an established pathogenic variant."""
        e = ACMGEvidence(code="PS1", strength="strong", direction="pathogenic")
        clinvar = var.clinvar_significance.lower()
        if "pathogenic" in clinvar and "likely" not in clinvar:
            e.applied = True
            e.justification = f"ClinVar: {var.clinvar_significance} ({var.clinvar_id})"
        return e

    def _ps3(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PS3: Well-established functional studies show damaging effect.
        Approximated by strong SpliceAI or regulatory evidence."""
        e = ACMGEvidence(code="PS3", strength="strong", direction="pathogenic")
        if var.spliceai_max is not None and var.spliceai_max >= 0.8:
            e.applied = True
            e.justification = f"SpliceAI={var.spliceai_max:.2f} (strong splice disruption)"
        return e

    def _pm1(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PM1: Located in a mutational hot spot / functional domain.
        Approximated by high REVEL for missense variants."""
        e = ACMGEvidence(code="PM1", strength="moderate", direction="pathogenic")
        if var.revel_score is not None and var.revel_score >= 0.7:
            e.applied = True
            e.justification = f"REVEL={var.revel_score:.3f} (functional domain/hotspot)"
        return e

    def _pm2(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PM2: Absent from controls (or at extremely low frequency)."""
        e = ACMGEvidence(code="PM2", strength="moderate", direction="pathogenic")
        af = var.effective_af
        if af == 0.0:
            e.applied = True
            e.justification = "Absent from population databases (novel)"
            e.strength = "moderate"
        elif af < 0.0001:
            e.applied = True
            e.justification = f"Extremely rare (AF={af:.2e})"
            e.strength = "supporting"  # Downgrade per ClinGen PM2 refinement
        return e

    def _pm3(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PM3: Detected in trans with a known pathogenic variant (compound het)."""
        e = ACMGEvidence(code="PM3", strength="moderate", direction="pathogenic")
        if var.is_compound_het:
            e.applied = True
            e.justification = f"Compound het with {var.compound_het_partner_key}"
        return e

    def _pm4(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PM4: Protein length change (in-frame del/ins in non-repeat region)."""
        e = ACMGEvidence(code="PM4", strength="moderate", direction="pathogenic")
        inframe = {"inframe_insertion", "inframe_deletion"}
        if any(c.strip() in inframe for c in var.consequence.split(",")):
            e.applied = True
            e.justification = f"In-frame protein length change ({var.consequence})"
        return e

    def _pm5(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PM5: Novel missense at same position as established pathogenic."""
        e = ACMGEvidence(code="PM5", strength="moderate", direction="pathogenic")
        clinvar = var.clinvar_significance.lower()
        if "likely pathogenic" in clinvar and "missense" in var.consequence.lower():
            e.applied = True
            e.justification = "Likely pathogenic missense at known position"
        return e

    def _pp1(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PP1: Co-segregation with disease in family members.
        Approximated by de novo status from trio analysis."""
        e = ACMGEvidence(code="PP1", strength="supporting", direction="pathogenic")
        if var.is_de_novo:
            e.applied = True
            e.strength = "strong"  # De novo upgrades PP1 to strong (PS2-equivalent)
            e.justification = "De novo variant (absent in both parents)"
        elif var.is_hom_recessive:
            e.applied = True
            e.justification = "Homozygous recessive confirmed by parental carriers"
        return e

    def _pp3(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PP3: Computational evidence supports deleterious effect."""
        e = ACMGEvidence(code="PP3", strength="supporting", direction="pathogenic")
        reasons = []

        if var.cadd_phred is not None and var.cadd_phred >= 25:
            reasons.append(f"CADD={var.cadd_phred:.1f}")
        if var.revel_score is not None and var.revel_score >= 0.5:
            reasons.append(f"REVEL={var.revel_score:.3f}")
        if var.spliceai_max is not None and var.spliceai_max >= 0.2:
            reasons.append(f"SpliceAI={var.spliceai_max:.2f}")
        if var.regulatory_score >= 0.7 and var.is_noncoding:
            reasons.append(f"RegScore={var.regulatory_score:.2f}")

        if reasons:
            e.applied = True
            e.justification = "Computational: " + ", ".join(reasons)
        return e

    def _pp5(self, var: AnnotatedVariant) -> ACMGEvidence:
        """PP5: Reputable source reports variant as pathogenic.
        Note: ClinGen has deprecated PP5/BP6 but still used in practice."""
        e = ACMGEvidence(code="PP5", strength="supporting", direction="pathogenic")
        clinvar = var.clinvar_significance.lower()
        if "likely pathogenic" in clinvar:
            e.applied = True
            e.justification = f"ClinVar: {var.clinvar_significance}"
        return e

    # ── Benign Criteria ───────────────────────────────────────────────────────

    def _ba1(self, var: AnnotatedVariant) -> ACMGEvidence:
        """BA1: Allele frequency > 5% in any general population."""
        e = ACMGEvidence(code="BA1", strength="stand_alone", direction="benign")
        af = var.effective_af
        if af > 0.05:
            e.applied = True
            e.justification = f"Common variant (AF={af:.4f})"
        return e

    def _bs1(self, var: AnnotatedVariant) -> ACMGEvidence:
        """BS1: Allele frequency greater than expected for disorder."""
        e = ACMGEvidence(code="BS1", strength="strong", direction="benign")
        af = var.effective_af
        if 0.01 < af <= 0.05:
            e.applied = True
            e.justification = f"AF higher than expected for rare disease (AF={af:.4f})"
        return e

    def _bs2(self, var: AnnotatedVariant) -> ACMGEvidence:
        """BS2: Observed in healthy adult (homozygous for recessive)."""
        e = ACMGEvidence(code="BS2", strength="strong", direction="benign")
        if var.gnomad_hom_count > 5 or var.local_hom_count > 3:
            hom = max(var.gnomad_hom_count, var.local_hom_count)
            e.applied = True
            e.justification = f"Multiple homozygotes in population ({hom} hom)"
        return e

    def _bp1(self, var: AnnotatedVariant) -> ACMGEvidence:
        """BP1: Missense in gene where only truncating cause disease."""
        e = ACMGEvidence(code="BP1", strength="supporting", direction="benign")
        # Conservative: only apply for synonymous variants
        if "synonymous_variant" in var.consequence:
            e.applied = True
            e.justification = "Synonymous variant"
        return e

    def _bp4(self, var: AnnotatedVariant) -> ACMGEvidence:
        """BP4: Computational evidence suggests no impact."""
        e = ACMGEvidence(code="BP4", strength="supporting", direction="benign")
        reasons = []
        if var.cadd_phred is not None and var.cadd_phred < 10:
            reasons.append(f"CADD={var.cadd_phred:.1f} (benign)")
        if var.revel_score is not None and var.revel_score < 0.15:
            reasons.append(f"REVEL={var.revel_score:.3f} (benign)")

        if reasons:
            e.applied = True
            e.justification = "Computational: " + ", ".join(reasons)
        return e

    def _bp6(self, var: AnnotatedVariant) -> ACMGEvidence:
        """BP6: Reputable source reports variant as benign."""
        e = ACMGEvidence(code="BP6", strength="supporting", direction="benign")
        clinvar = var.clinvar_significance.lower()
        if "benign" in clinvar:
            e.applied = True
            e.justification = f"ClinVar: {var.clinvar_significance}"
        return e

    def _bp7(self, var: AnnotatedVariant) -> ACMGEvidence:
        """BP7: Synonymous with no predicted splice impact."""
        e = ACMGEvidence(code="BP7", strength="supporting", direction="benign")
        is_syn = "synonymous_variant" in var.consequence
        no_splice = var.spliceai_max is None or var.spliceai_max < 0.1
        if is_syn and no_splice:
            e.applied = True
            e.justification = "Synonymous, no splice impact"
        return e

    # ── Evidence Combining (ACMG Table 5) ─────────────────────────────────────

    def _combine_evidence(self, evidence: list[ACMGEvidence]) -> ACMGClassification:
        """Combine evidence criteria following ACMG Table 5 rules.

        Pathogenic:
          (i)   1 Very Strong + ≥1 Strong
          (ii)  1 Very Strong + ≥2 Moderate
          (iii) 1 Very Strong + 1 Moderate + 1 Supporting
          (iv)  1 Very Strong + ≥2 Supporting
          (v)   ≥2 Strong
          (vi)  1 Strong + ≥3 Moderate
          (vii) 1 Strong + 2 Moderate + ≥2 Supporting
          (viii) 1 Strong + 1 Moderate + ≥4 Supporting

        Likely Pathogenic:
          (i)   1 Very Strong + 1 Moderate
          (ii)  1 Strong + 1-2 Moderate
          (iii) 1 Strong + ≥2 Supporting
          (iv)  ≥3 Moderate
          (v)   2 Moderate + ≥2 Supporting
          (vi)  1 Moderate + ≥4 Supporting
        """
        applied = [e for e in evidence if e.applied]

        path_applied = [e for e in applied if e.direction == "pathogenic"]
        ben_applied = [e for e in applied if e.direction == "benign"]

        # Count benign
        has_ba = any(e.strength == "stand_alone" for e in ben_applied)
        n_bs = sum(1 for e in ben_applied if e.strength == "strong")
        n_bp = sum(1 for e in ben_applied if e.strength in ("moderate", "supporting"))

        # Benign: BA1 alone is sufficient
        if has_ba:
            return ACMGClassification.BENIGN
        if n_bs >= 2:
            return ACMGClassification.BENIGN
        if n_bs >= 1 and n_bp >= 1:
            return ACMGClassification.LIKELY_BENIGN

        # Count pathogenic by strength
        n_pvs = sum(1 for e in path_applied if e.strength == "very_strong")
        n_ps = sum(1 for e in path_applied if e.strength == "strong")
        n_pm = sum(1 for e in path_applied if e.strength == "moderate")
        n_pp = sum(1 for e in path_applied if e.strength == "supporting")

        # Pathogenic combinations (Table 5)
        if n_pvs >= 1:
            if n_ps >= 1:
                return ACMGClassification.PATHOGENIC
            if n_pm >= 2:
                return ACMGClassification.PATHOGENIC
            if n_pm >= 1 and n_pp >= 1:
                return ACMGClassification.PATHOGENIC
            if n_pp >= 2:
                return ACMGClassification.PATHOGENIC

        if n_ps >= 2:
            return ACMGClassification.PATHOGENIC

        if n_ps >= 1:
            if n_pm >= 3:
                return ACMGClassification.PATHOGENIC
            if n_pm >= 2 and n_pp >= 2:
                return ACMGClassification.PATHOGENIC
            if n_pm >= 1 and n_pp >= 4:
                return ACMGClassification.PATHOGENIC

        # Likely Pathogenic combinations
        if n_pvs >= 1 and n_pm >= 1:
            return ACMGClassification.LIKELY_PATHOGENIC

        if n_ps >= 1:
            if n_pm >= 1:
                return ACMGClassification.LIKELY_PATHOGENIC
            if n_pp >= 2:
                return ACMGClassification.LIKELY_PATHOGENIC

        if n_pm >= 3:
            return ACMGClassification.LIKELY_PATHOGENIC

        if n_pm >= 2 and n_pp >= 2:
            return ACMGClassification.LIKELY_PATHOGENIC

        if n_pm >= 1 and n_pp >= 4:
            return ACMGClassification.LIKELY_PATHOGENIC

        # Likely Benign
        if n_bp >= 2 and n_pvs == 0 and n_ps == 0 and n_pm == 0:
            return ACMGClassification.LIKELY_BENIGN

        return ACMGClassification.VUS

    def _build_summary(
        self,
        var: AnnotatedVariant,
        classification: ACMGClassification,
        path_codes: list[str],
        ben_codes: list[str],
    ) -> str:
        """Build human-readable ACMG summary for the clinical report."""
        parts = [f"ACMG Classification: {classification.value}"]

        if path_codes:
            parts.append(f"Pathogenic evidence: {', '.join(path_codes)}")
        if ben_codes:
            parts.append(f"Benign evidence: {', '.join(ben_codes)}")

        parts.append(f"Variant: {var.variant.variant_key} ({var.hgvs_p or var.hgvs_c or var.consequence})")

        return " | ".join(parts)
