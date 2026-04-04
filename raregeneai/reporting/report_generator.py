"""Clinical report generator.

Produces ACMG-aligned clinical reports in HTML/PDF format.
Designed for clinical geneticists and diagnostic laboratories.
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path

from loguru import logger

from raregeneai.config.settings import ReportConfig
from raregeneai.explainability.explainer import Explainer
from raregeneai.models.data_models import (
    ClinicalReport,
    GeneCandidate,
    PatientPhenotype,
    Pedigree,
)


class ReportGenerator:
    """Generate clinical genomics reports."""

    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()
        self.explainer = Explainer()

    def generate(
        self,
        patient_phenotype: PatientPhenotype,
        ranked_genes: list[GeneCandidate],
        total_variants: int = 0,
        total_genes: int = 0,
        pedigree: Pedigree | None = None,
        output_path: str | None = None,
    ) -> ClinicalReport:
        """Generate a complete clinical report.

        Args:
            patient_phenotype: Patient HPO phenotype profile.
            ranked_genes: Prioritized gene list.
            total_variants: Total variants analyzed.
            total_genes: Total genes analyzed.
            pedigree: Optional family pedigree.
            output_path: Where to save the report.

        Returns:
            ClinicalReport object with rendered content.
        """
        report = ClinicalReport(
            patient_id=patient_phenotype.patient_id,
            report_id=str(uuid.uuid4())[:8],
            report_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            phenotype=patient_phenotype,
            pedigree=pedigree,
            ranked_genes=ranked_genes,
            total_variants_analyzed=total_variants,
            total_genes_analyzed=total_genes,
        )

        # Generate explanations for each gene
        for gene in ranked_genes[:20]:
            self.explainer.explain_gene(gene, patient_phenotype)

        # Render report
        if output_path:
            output_path = Path(output_path)
            if self.config.output_format == "html":
                self._render_html(report, output_path)
            elif self.config.output_format == "pdf":
                self._render_pdf(report, output_path)

            logger.info(f"Report saved to {output_path}")

        return report

    def _render_html(self, report: ClinicalReport, output_path: Path) -> None:
        """Render report as HTML."""
        try:
            from jinja2 import Environment, FileSystemLoader

            template_dir = Path(self.config.template_dir)
            if template_dir.exists() and (template_dir / "report.html").exists():
                env = Environment(loader=FileSystemLoader(str(template_dir)))
                template = env.get_template("report.html")
            else:
                template = Environment().from_string(self._default_html_template())

            html = template.render(
                report=report,
                lab_name=self.config.lab_name,
                report_version=self.config.report_version,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html)

        except ImportError:
            # Fallback: write raw HTML without Jinja
            html = self._render_html_fallback(report)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html)

    def _render_pdf(self, report: ClinicalReport, output_path: Path) -> None:
        """Render report as PDF via HTML intermediate."""
        html_path = output_path.with_suffix(".html")
        self._render_html(report, html_path)

        try:
            from weasyprint import HTML

            HTML(filename=str(html_path)).write_pdf(str(output_path))
            logger.info("PDF generated with WeasyPrint")
        except ImportError:
            logger.warning("WeasyPrint not installed; HTML report generated instead")

    def _render_html_fallback(self, report: ClinicalReport) -> str:
        """Generate HTML report without Jinja2."""
        genes_html = ""
        for i, gene in enumerate(report.ranked_genes[:20], 1):
            variants_html = ""
            for var in gene.variants[:5]:
                acmg = self.explainer.classify_acmg(var)
                variants_html += f"""
                <tr>
                    <td>{var.variant.variant_key}</td>
                    <td>{var.hgvs_p or var.hgvs_c or '-'}</td>
                    <td>{var.consequence}</td>
                    <td>{var.impact.value}</td>
                    <td>{f'{var.gnomad_af:.2e}' if var.gnomad_af else 'Novel'}</td>
                    <td>{f'{var.cadd_phred:.1f}' if var.cadd_phred else '-'}</td>
                    <td>{var.clinvar_significance or '-'}</td>
                    <td>{acmg.value}</td>
                </tr>"""

            # Evidence badges
            badges_html = ""
            if gene.has_de_novo_lof:
                badges_html += '<span style="background:#c0392b;color:white;padding:2px 6px;border-radius:3px;font-size:0.75em;margin-right:4px;">DE NOVO LoF</span>'
            if gene.has_sv:
                badges_html += '<span style="background:#8e44ad;color:white;padding:2px 6px;border-radius:3px;font-size:0.75em;margin-right:4px;">SV</span>'
            if gene.multi_omics_score > 0.3:
                badges_html += '<span style="background:#2980b9;color:white;padding:2px 6px;border-radius:3px;font-size:0.75em;margin-right:4px;">Multi-omics</span>'
            if gene.kg_score > 0.1:
                badges_html += '<span style="background:#27ae60;color:white;padding:2px 6px;border-radius:3px;font-size:0.75em;margin-right:4px;">KG:{gene.kg_score:.2f}</span>'
            ev = gene.evidence_summary
            if ev.get("is_acmg_sf_gene"):
                badges_html += '<span style="background:#e67e22;color:white;padding:2px 6px;border-radius:3px;font-size:0.75em;margin-right:4px;">ACMG SF</span>'
            if ev.get("has_pgx_relevance"):
                badges_html += '<span style="background:#16a085;color:white;padding:2px 6px;border-radius:3px;font-size:0.75em;margin-right:4px;">PGx</span>'
            if ev.get("is_founder_variant"):
                badges_html += '<span style="background:#d35400;color:white;padding:2px 6px;border-radius:3px;font-size:0.75em;">Founder</span>'

            genes_html += f"""
            <div class="gene-card">
                <h3>#{i}. {gene.gene_symbol}
                    <span class="score">Score: {gene.gene_rank_score:.3f}</span>
                    <span class="confidence">Confidence: {gene.confidence:.0%}</span>
                </h3>
                <div style="margin-bottom:8px;">{badges_html}</div>
                <p class="phenotype-score">Phenotype match: {gene.phenotype_score:.3f}
                    | Inheritance: {gene.inheritance_score:.2f}
                    | Evidence layers: {gene.n_evidence_layers}</p>
                <pre class="explanation">{gene.explanation}</pre>
                <table>
                    <thead>
                        <tr>
                            <th>Variant</th><th>Protein</th><th>Consequence</th>
                            <th>Impact</th><th>gnomAD AF</th><th>CADD</th>
                            <th>ClinVar</th><th>ACMG</th>
                        </tr>
                    </thead>
                    <tbody>{variants_html}</tbody>
                </table>
            </div>"""

        hpo_list = ", ".join(
            f"{t.id} ({t.name})" for t in report.phenotype.hpo_terms
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RareGeneAI Clinical Report - {report.patient_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }}
        .header {{ background: #1a5276; color: white; padding: 20px; border-radius: 8px; }}
        .header h1 {{ margin: 0; }}
        .meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 20px 0;
                  background: #f8f9fa; padding: 15px; border-radius: 8px; }}
        .gene-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px;
                      margin: 15px 0; }}
        .gene-card h3 {{ color: #1a5276; margin-top: 0; }}
        .score {{ background: #e74c3c; color: white; padding: 2px 8px; border-radius: 4px;
                  font-size: 0.8em; margin-left: 10px; }}
        .confidence {{ background: #27ae60; color: white; padding: 2px 8px; border-radius: 4px;
                       font-size: 0.8em; }}
        .phenotype-score {{ color: #7f8c8d; }}
        .explanation {{ background: #f0f0f0; padding: 10px; border-radius: 4px;
                        font-size: 0.85em; white-space: pre-wrap; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.85em; }}
        th {{ background: #34495e; color: white; padding: 8px; text-align: left; }}
        td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f5f5f5; }}
        .footer {{ margin-top: 30px; padding: 15px; background: #f8f9fa;
                   border-radius: 8px; font-size: 0.85em; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RareGeneAI Clinical Report</h1>
        <p>{self.config.lab_name}</p>
    </div>

    <div class="meta">
        <div><strong>Patient ID:</strong> {report.patient_id}</div>
        <div><strong>Report ID:</strong> {report.report_id}</div>
        <div><strong>Report Date:</strong> {report.report_date}</div>
        <div><strong>Genome Build:</strong> {report.genome_build}</div>
        <div><strong>Variants Analyzed:</strong> {report.total_variants_analyzed}</div>
        <div><strong>SVs Analyzed:</strong> {report.total_svs_analyzed}</div>
        <div><strong>Genes Analyzed:</strong> {report.total_genes_analyzed}</div>
    </div>

    <h2>Clinical Phenotype</h2>
    <p><strong>HPO Terms:</strong> {hpo_list}</p>
    <p><strong>Sex:</strong> {report.phenotype.sex or 'Not specified'}</p>
    <p><strong>Age of Onset:</strong> {report.phenotype.age_of_onset or 'Not specified'}</p>

    <h2>Candidate Genes (Ranked)</h2>
    {genes_html}

    <div class="footer">
        <p><strong>Disclaimer:</strong> This report is generated by RareGeneAI v{report.pipeline_version}
        and is intended to support clinical decision-making. Findings should be confirmed through
        independent methods and interpreted by qualified clinical geneticists.</p>
        <p>Report version: {self.config.report_version} | Pipeline: RareGeneAI v{report.pipeline_version}</p>
    </div>
</body>
</html>"""

    def _default_html_template(self) -> str:
        """Jinja2 template for HTML report."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RareGeneAI Report - {{ report.patient_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #1a5276; color: white; padding: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #34495e; color: white; padding: 8px; }
        td { padding: 6px; border-bottom: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RareGeneAI Clinical Report</h1>
        <p>{{ lab_name }}</p>
    </div>
    <p>Patient: {{ report.patient_id }} | Date: {{ report.report_date }}</p>
    {% for gene in report.ranked_genes[:20] %}
    <h3>#{{ loop.index }}. {{ gene.gene_symbol }} (Score: {{ "%.3f"|format(gene.gene_rank_score) }})</h3>
    <pre>{{ gene.explanation }}</pre>
    {% endfor %}
</body>
</html>"""
