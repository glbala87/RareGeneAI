"""RareGeneAI command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from raregeneai.config.settings import PipelineConfig
from raregeneai.pipeline.orchestrator import RareGeneAIPipeline

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="RareGeneAI")
def main():
    """RareGeneAI - Rare Disease Gene Prioritization System."""
    pass


@main.command()
@click.option("--vcf", required=True, type=click.Path(exists=True), help="Input VCF file")
@click.option("--hpo", required=True, multiple=True, help="HPO term(s), e.g., HP:0001250")
@click.option("--patient-id", default="PATIENT_001", help="Patient identifier")
@click.option("--sample-id", default=None, help="VCF sample ID")
@click.option("--ped", default=None, type=click.Path(exists=True), help="PED file for trio")
@click.option("--sv-vcf", default=None, type=click.Path(exists=True), help="SV VCF (Sniffles/Jasmine)")
@click.option("--father-vcf", default=None, type=click.Path(exists=True), help="Father VCF for trio analysis")
@click.option("--mother-vcf", default=None, type=click.Path(exists=True), help="Mother VCF for trio analysis")
@click.option("--expression", default=None, type=click.Path(exists=True), help="RNA-seq TPM file")
@click.option("--methylation", default=None, type=click.Path(exists=True), help="Methylation BED or DMR calls")
@click.option("--output", "-o", default="output", help="Output directory")
@click.option("--config", "-c", default=None, type=click.Path(exists=True), help="Config YAML")
@click.option("--top-n", default=20, help="Number of top genes to report")
def analyze(vcf, hpo, patient_id, sample_id, ped, sv_vcf, father_vcf, mother_vcf, expression, methylation, output, config, top_n):
    """Run the full gene prioritization pipeline."""
    console.print("[bold blue]RareGeneAI[/bold blue] - Rare Disease Gene Prioritization")
    console.print()

    # Load config
    if config:
        pipeline_config = PipelineConfig.from_yaml(config)
    else:
        pipeline_config = PipelineConfig()

    pipeline_config.ranking.top_n_genes = top_n

    # Run pipeline
    pipeline = RareGeneAIPipeline(pipeline_config)
    report = pipeline.run(
        vcf_path=vcf,
        hpo_terms=list(hpo),
        patient_id=patient_id,
        sample_id=sample_id,
        ped_path=ped,
        sv_vcf_path=sv_vcf,
        father_vcf_path=father_vcf,
        mother_vcf_path=mother_vcf,
        expression_path=expression,
        methylation_path=methylation,
        output_dir=output,
    )

    # Display results
    table = Table(title=f"Top {top_n} Candidate Genes")
    table.add_column("Rank", style="bold")
    table.add_column("Gene", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("Phenotype", style="magenta")
    table.add_column("Variants", style="white")
    table.add_column("Top Evidence", style="white")

    for i, gene in enumerate(report.ranked_genes[:top_n], 1):
        top_evidence = []
        if gene.has_lof_variant:
            top_evidence.append("LoF")
        if any("pathogenic" in v.clinvar_significance.lower() for v in gene.variants):
            top_evidence.append("ClinVar:P")
        if gene.phenotype_score > 0.5:
            top_evidence.append("HPO-match")

        table.add_row(
            str(i),
            gene.gene_symbol,
            f"{gene.gene_rank_score:.3f}",
            f"{gene.confidence:.0%}",
            f"{gene.phenotype_score:.3f}",
            str(gene.n_variants),
            ", ".join(top_evidence) or "-",
        )

    console.print(table)
    console.print(f"\n[green]Report saved to:[/green] {output}/{patient_id}_report.html")


@main.command()
@click.option("--output", "-o", default="config.yaml", help="Output config path")
def init_config(output):
    """Generate a default configuration file."""
    config = PipelineConfig()
    config.to_yaml(output)
    console.print(f"[green]Config saved to {output}[/green]")


@main.command()
def ui():
    """Launch the Streamlit web interface."""
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])


if __name__ == "__main__":
    main()
