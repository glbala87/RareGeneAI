"""SV-to-SNV integration bridge for unified gene ranking.

Converts AnnotatedSV objects into AnnotatedVariant-compatible representations
so they can be merged with SNV/indel variants in the unified gene ranking.

Also provides direct GeneCandidate enrichment for SV evidence.
"""

from __future__ import annotations

from loguru import logger

from raregeneai.models.data_models import (
    AnnotatedSV,
    AnnotatedVariant,
    FunctionalImpact,
    GeneCandidate,
    StructuralVariant,
    SVType,
    Variant,
    Zygosity,
)


# Map SV types to VEP-style consequence terms
_SV_CONSEQUENCE_MAP = {
    SVType.DEL: "deletion",
    SVType.DUP: "duplication",
    SVType.INV: "inversion",
    SVType.INS: "insertion",
    SVType.BND: "translocation",
    SVType.CNV: "copy_number_variation",
    SVType.DUP_TANDEM: "tandem_duplication",
    SVType.DEL_ME: "mobile_element_deletion",
    SVType.INS_ME: "mobile_element_insertion",
}


def sv_to_annotated_variants(
    annotated_svs: list[AnnotatedSV],
) -> list[AnnotatedVariant]:
    """Convert AnnotatedSV objects to AnnotatedVariant for unified ranking.

    Each SV generates one AnnotatedVariant per overlapping gene.
    The SV score is carried through as the composite_score.
    """
    variants = []

    for ann in annotated_svs:
        sv = ann.sv

        for gene in ann.overlapping_genes:
            # Create a pseudo-Variant for the SV
            base_variant = Variant(
                chrom=sv.chrom,
                pos=sv.pos,
                ref="N",
                alt=f"<{sv.sv_type.value}>",
                variant_id=sv.sv_id,
                quality=sv.quality,
                zygosity=sv.zygosity,
                genotype=sv.genotype,
                depth=sv.support_reads + sv.ref_reads,
                gq=int(min(sv.quality, 99)),
                allele_depth_ref=sv.ref_reads,
                allele_depth_alt=sv.support_reads,
                sample_id=sv.sample_id,
            )

            # Determine impact level
            is_fully_deleted = gene in ann.fully_deleted_genes
            impact = _sv_impact(sv, is_fully_deleted)

            # Build consequence string
            consequence = _SV_CONSEQUENCE_MAP.get(sv.sv_type, "structural_variant")
            if is_fully_deleted:
                consequence = f"whole_gene_{consequence}"

            # Compute effective gnomAD AF from SV frequency
            gnomad_af = ann.gnomad_sv_af or ann.dgv_frequency

            av = AnnotatedVariant(
                variant=base_variant,
                gene_symbol=gene,
                gene_id="",
                consequence=consequence,
                impact=impact,
                gnomad_af=gnomad_af,
                clinvar_significance=ann.clinvar_sv_significance,
                # Carry SV scores through existing fields
                cadd_phred=None,
                revel_score=None,
                spliceai_max=None,
                # Use noncoding_impact_score to carry regulatory disruption
                noncoding_impact_score=ann.regulatory_disruption_score,
                # Pre-compute scores from SV annotation
                pathogenicity_score=ann.sv_pathogenicity_score,
                rarity_score=ann.sv_rarity_score,
                functional_score=ann.gene_overlap_score,
                composite_score=ann.sv_composite_score,
            )

            variants.append(av)

    n_genes = len({v.gene_symbol for v in variants})
    logger.info(
        f"Converted {len(annotated_svs)} SVs to {len(variants)} "
        f"gene-level variant records across {n_genes} genes"
    )
    return variants


def enrich_candidates_with_sv(
    candidates: list[GeneCandidate],
    annotated_svs: list[AnnotatedSV],
) -> list[GeneCandidate]:
    """Enrich GeneCandidate objects with SV-specific evidence.

    Adds SV evidence fields and adjusts scores for genes affected by SVs.
    """
    # Build gene -> SV lookup
    gene_svs: dict[str, list[AnnotatedSV]] = {}
    for ann in annotated_svs:
        for gene in ann.overlapping_genes:
            gene_svs.setdefault(gene, []).append(ann)

    for candidate in candidates:
        svs = gene_svs.get(candidate.gene_symbol, [])
        if not svs:
            continue

        candidate.structural_variants = svs
        candidate.has_sv = True
        candidate.max_sv_score = max(sv.sv_composite_score for sv in svs)
        candidate.sv_fully_deleted = any(
            candidate.gene_symbol in sv.fully_deleted_genes for sv in svs
        )
        candidate.sv_dosage_sensitive = any(sv.has_dosage_sensitive_gene for sv in svs)

        # Update evidence_summary with SV features
        candidate.evidence_summary.update({
            "has_sv": True,
            "max_sv_score": candidate.max_sv_score,
            "n_svs": len(svs),
            "sv_fully_deleted": candidate.sv_fully_deleted,
            "sv_dosage_sensitive": candidate.sv_dosage_sensitive,
            "max_sv_gene_overlap": max(sv.gene_overlap_score for sv in svs),
            "max_sv_dosage_score": max(sv.dosage_sensitivity_score for sv in svs),
            "max_sv_regulatory_disruption": max(
                sv.regulatory_disruption_score for sv in svs
            ),
        })

    n_enriched = sum(1 for c in candidates if c.has_sv)
    logger.info(f"Enriched {n_enriched} gene candidates with SV evidence")
    return candidates


def _sv_impact(sv: StructuralVariant, fully_deleted: bool) -> FunctionalImpact:
    """Determine functional impact of an SV on a gene."""
    if fully_deleted:
        return FunctionalImpact.HIGH

    if sv.is_deletion:
        return FunctionalImpact.HIGH  # Partial gene deletion

    if sv.sv_type == SVType.BND:
        return FunctionalImpact.HIGH  # Translocation disrupting gene

    if sv.is_duplication:
        return FunctionalImpact.MODERATE  # Duplication may cause dosage change

    if sv.sv_type == SVType.INV:
        return FunctionalImpact.MODERATE  # Inversion may disrupt gene

    if sv.sv_type == SVType.INS:
        return FunctionalImpact.MODERATE

    return FunctionalImpact.MODIFIER
