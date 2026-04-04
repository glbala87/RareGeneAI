"""RareGeneAI Streamlit Web Interface.

Provides interactive UI for:
  - VCF file upload
  - HPO term entry with auto-complete
  - Gene ranking visualization
  - Interactive filtering and exploration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="RareGeneAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("🧬 RareGeneAI")
    st.markdown("**Rare Disease Gene Prioritization System**")
    st.markdown("---")

    # ── Sidebar: Configuration ─────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        genome_build = st.selectbox("Genome Build", ["GRCh38", "GRCh37"])
        ranking_model = st.selectbox("Ranking Model", ["rule_based", "xgboost", "lightgbm"])
        top_n = st.slider("Top N genes", 5, 100, 20)

        st.markdown("---")
        st.header("Scoring Weights")
        w_patho = st.slider("Pathogenicity", 0.0, 1.0, 0.35, 0.05)
        w_rarity = st.slider("Rarity", 0.0, 1.0, 0.25, 0.05)
        w_impact = st.slider("Functional Impact", 0.0, 1.0, 0.20, 0.05)
        w_pheno = st.slider("Phenotype Match", 0.0, 1.0, 0.15, 0.05)
        w_inherit = st.slider("Inheritance", 0.0, 1.0, 0.05, 0.05)

        st.markdown("---")
        st.header("Filters")
        gnomad_threshold = st.number_input("gnomAD AF threshold", value=0.01, format="%.4f")
        cadd_threshold = st.number_input("Min CADD PHRED", value=15.0)
        require_rare = st.checkbox("Require rare variants", value=True)
        require_coding = st.checkbox("Coding variants only", value=False)

    # ── Main: Input ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 VCF Input")
        vcf_file = st.file_uploader(
            "Upload VCF file",
            type=["vcf", "vcf.gz", "gz"],
            help="WGS or WES VCF file",
        )
        sample_id = st.text_input("Sample ID (optional)")
        patient_id = st.text_input("Patient ID", value="PATIENT_001")

    with col2:
        st.subheader("🏥 Phenotype (HPO Terms)")
        hpo_input = st.text_area(
            "Enter HPO terms (one per line)",
            placeholder="HP:0001250\nHP:0002878\nHP:0001263",
            height=120,
        )
        st.markdown("**Common HPO terms:**")
        common_terms = {
            "Seizures": "HP:0001250",
            "Intellectual disability": "HP:0001249",
            "Microcephaly": "HP:0000252",
            "Hypotonia": "HP:0001252",
            "Short stature": "HP:0004322",
            "Ataxia": "HP:0001251",
        }
        selected_common = st.multiselect("Quick add", list(common_terms.keys()))

        ped_file = st.file_uploader("PED file (optional)", type=["ped", "txt"])

    # ── Additional Data Sources ────────────────────────────────
    st.markdown("---")
    st.subheader("Additional Data (Optional)")
    col3, col4, col5 = st.columns(3)

    with col3:
        sv_vcf_file = st.file_uploader(
            "SV VCF (Sniffles/Manta)",
            type=["vcf", "vcf.gz", "gz"],
            help="Structural variant calls",
        )
        father_vcf_file = st.file_uploader("Father VCF (trio)", type=["vcf", "vcf.gz", "gz"])
        mother_vcf_file = st.file_uploader("Mother VCF (trio)", type=["vcf", "vcf.gz", "gz"])

    with col4:
        expression_file = st.file_uploader(
            "RNA-seq expression (TPM)",
            type=["tsv", "txt", "csv"],
            help="Gene expression: gene<TAB>tpm",
        )
        methylation_file = st.file_uploader(
            "Methylation data / DMR calls",
            type=["tsv", "txt", "bed", "csv"],
            help="BED beta values or pre-called DMRs",
        )

    with col5:
        population = st.selectbox(
            "Patient population",
            ["Auto-detect", "QGP", "GME", "European", "African", "South Asian", "East Asian", "Other"],
        )

    # Combine HPO terms
    hpo_terms = []
    if hpo_input:
        for line in hpo_input.strip().split("\n"):
            term = line.strip()
            if term.startswith("HP:"):
                hpo_terms.append(term)
    for name in selected_common:
        hpo_terms.append(common_terms[name])
    hpo_terms = list(set(hpo_terms))

    if hpo_terms:
        st.info(f"**{len(hpo_terms)} HPO terms selected:** {', '.join(hpo_terms)}")

    # ── Run Analysis ───────────────────────────────────────────────
    st.markdown("---")

    if st.button("🚀 Run Gene Prioritization", type="primary", use_container_width=True):
        if not vcf_file:
            st.error("Please upload a VCF file.")
            return
        if not hpo_terms:
            st.error("Please enter at least one HPO term.")
            return

        with st.spinner("Running RareGeneAI pipeline..."):
            try:
                from raregeneai.config.settings import PipelineConfig
                from raregeneai.pipeline.orchestrator import RareGeneAIPipeline

                # Save uploaded VCF to temp
                with tempfile.NamedTemporaryFile(
                    suffix=".vcf.gz" if vcf_file.name.endswith(".gz") else ".vcf",
                    delete=False,
                ) as tmp:
                    tmp.write(vcf_file.getvalue())
                    tmp_vcf_path = tmp.name

                # Save optional files to temp
                def _save_upload(upload, suffix):
                    if upload is None:
                        return None
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as t:
                        t.write(upload.getvalue())
                        return t.name

                tmp_ped_path = _save_upload(ped_file, ".ped")
                tmp_sv_path = _save_upload(sv_vcf_file, ".vcf")
                tmp_father_path = _save_upload(father_vcf_file, ".vcf")
                tmp_mother_path = _save_upload(mother_vcf_file, ".vcf")
                tmp_expr_path = _save_upload(expression_file, ".tsv")
                tmp_meth_path = _save_upload(methylation_file, ".tsv")

                # Build config
                config = PipelineConfig()
                config.scoring.w_pathogenicity = w_patho
                config.scoring.w_rarity = w_rarity
                config.scoring.w_functional_impact = w_impact
                config.scoring.w_phenotype = w_pheno
                config.scoring.w_inheritance = w_inherit
                config.scoring.gnomad_af_threshold = gnomad_threshold
                config.scoring.cadd_phred_threshold = cadd_threshold
                config.ranking.model_type = ranking_model
                config.ranking.top_n_genes = top_n
                config.genome_build = genome_build

                if population and population != "Auto-detect":
                    config.annotation.population.population = population

                # Run pipeline
                pipeline = RareGeneAIPipeline(config)
                report = pipeline.run(
                    vcf_path=tmp_vcf_path,
                    hpo_terms=hpo_terms,
                    patient_id=patient_id,
                    sample_id=sample_id or None,
                    ped_path=tmp_ped_path,
                    sv_vcf_path=tmp_sv_path,
                    father_vcf_path=tmp_father_path,
                    mother_vcf_path=tmp_mother_path,
                    expression_path=tmp_expr_path,
                    methylation_path=tmp_meth_path,
                )

                st.session_state["report"] = report
                st.success(f"Analysis complete! {len(report.ranked_genes)} genes ranked.")

            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # ── Display Results ────────────────────────────────────────────
    if "report" in st.session_state:
        report = st.session_state["report"]
        _display_results(report)


def _display_results(report):
    """Display pipeline results."""
    import pandas as pd

    st.markdown("---")
    st.header("📊 Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variants Analyzed", report.total_variants_analyzed)
    col2.metric("Genes Analyzed", report.total_genes_analyzed)
    col3.metric("Candidate Genes", len(report.ranked_genes))
    col4.metric(
        "Top Gene",
        report.ranked_genes[0].gene_symbol if report.ranked_genes else "N/A",
    )

    # Gene ranking table
    st.subheader("🏆 Ranked Candidate Genes")
    records = []
    for i, gene in enumerate(report.ranked_genes, 1):
        records.append({
            "Rank": i,
            "Gene": gene.gene_symbol,
            "Score": f"{gene.gene_rank_score:.3f}",
            "Confidence": f"{gene.confidence:.0%}",
            "Phenotype Score": f"{gene.phenotype_score:.3f}",
            "Variants": gene.n_variants,
            "Has LoF": "Yes" if gene.has_lof_variant else "No",
        })

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Gene detail expanders
    st.subheader("🔍 Gene Details")
    for gene in report.ranked_genes[:10]:
        with st.expander(
            f"**{gene.gene_symbol}** — Score: {gene.gene_rank_score:.3f} | "
            f"Confidence: {gene.confidence:.0%}"
        ):
            if gene.explanation:
                st.code(gene.explanation)

            if gene.variants:
                var_records = []
                for v in gene.variants:
                    var_records.append({
                        "Variant": v.variant.variant_key,
                        "Protein": v.hgvs_p or "-",
                        "Consequence": v.consequence,
                        "Impact": v.impact.value,
                        "gnomAD AF": f"{v.gnomad_af:.2e}" if v.gnomad_af else "Novel",
                        "CADD": f"{v.cadd_phred:.1f}" if v.cadd_phred else "-",
                        "ClinVar": v.clinvar_significance or "-",
                        "Score": f"{v.composite_score:.3f}",
                    })
                st.dataframe(pd.DataFrame(var_records), use_container_width=True, hide_index=True)

    # Download report
    st.markdown("---")
    st.subheader("📥 Download")

    report_json = json.dumps(
        {
            "patient_id": report.patient_id,
            "report_id": report.report_id,
            "ranked_genes": [
                {
                    "gene": g.gene_symbol,
                    "score": g.gene_rank_score,
                    "confidence": g.confidence,
                    "phenotype_score": g.phenotype_score,
                    "n_variants": g.n_variants,
                    "explanation": g.explanation,
                }
                for g in report.ranked_genes
            ],
        },
        indent=2,
    )

    st.download_button(
        "Download JSON Report",
        data=report_json,
        file_name=f"{report.patient_id}_raregeneai_report.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
