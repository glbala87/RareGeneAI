#!/usr/bin/env nextflow

/*
 * RareGeneAI - Nextflow DSL2 Pipeline
 * =====================================
 * Rare Disease Gene Prioritization Pipeline
 *
 * Usage:
 *   nextflow run main.nf \
 *     --vcf sample.vcf.gz \
 *     --hpo "HP:0001250,HP:0002878" \
 *     --outdir results/ \
 *     --genome GRCh38
 */

nextflow.enable.dsl = 2

// ── Parameters ────────────────────────────────────────────────────────────────
params.vcf        = null
params.hpo        = null
params.ped        = null
params.patient_id = "PATIENT_001"
params.outdir     = "results"
params.genome     = "GRCh38"
params.top_n      = 50

// Reference data
params.vep_cache     = null
params.gnomad_vcf    = null
params.clinvar_vcf   = null
params.cadd_snv      = null
params.hpo_obo       = null
params.gene_pheno    = null

// ── Input validation ──────────────────────────────────────────────────────────
if (!params.vcf) { exit 1, "Error: --vcf parameter is required" }
if (!params.hpo) { exit 1, "Error: --hpo parameter is required" }


// ── Processes ─────────────────────────────────────────────────────────────────

process NORMALIZE_VCF {
    tag "${sample_id}"
    container 'quay.io/biocontainers/bcftools:1.19--h8b25389_0'

    input:
    tuple val(sample_id), path(vcf)

    output:
    tuple val(sample_id), path("${sample_id}.normalized.vcf.gz"), path("${sample_id}.normalized.vcf.gz.tbi")

    script:
    """
    bcftools norm -m -any -f ${params.genome == 'GRCh38' ? '/ref/GRCh38.fa' : '/ref/GRCh37.fa'} \\
        -o ${sample_id}.normalized.vcf.gz -O z ${vcf} || \\
    bcftools norm -m -any -o ${sample_id}.normalized.vcf.gz -O z ${vcf}

    bcftools index -t ${sample_id}.normalized.vcf.gz
    """
}

process ANNOTATE_VEP {
    tag "${sample_id}"
    container 'ensemblorg/ensembl-vep:release_112.0'
    cpus 4
    memory '8 GB'

    input:
    tuple val(sample_id), path(vcf), path(tbi)

    output:
    tuple val(sample_id), path("${sample_id}.vep.vcf.gz")

    script:
    """
    vep --input_file ${vcf} \\
        --output_file ${sample_id}.vep.vcf.gz \\
        --format vcf --vcf --compress_output bgzip \\
        --assembly ${params.genome} \\
        --offline --cache --dir_cache ${params.vep_cache ?: '/opt/vep/.vep'} \\
        --everything --fork ${task.cpus} \\
        --force_overwrite --no_stats
    """
}

process ANNOTATE_FREQUENCIES {
    tag "${sample_id}"
    container 'raregeneai/annotator:latest'
    cpus 2
    memory '4 GB'

    input:
    tuple val(sample_id), path(vep_vcf)

    output:
    tuple val(sample_id), path("${sample_id}.annotated.parquet")

    script:
    """
    python3 -c "
from raregeneai.ingestion.vcf_parser import VCFParser
from raregeneai.annotation.annotation_engine import AnnotationEngine
from raregeneai.config.settings import AnnotationConfig
import pandas as pd

parser = VCFParser()
variants = parser.parse('${vep_vcf}')

config = AnnotationConfig()
engine = AnnotationEngine(config)
annotated = engine.annotate_variants(variants)
df = engine.to_dataframe(annotated)
df.to_parquet('${sample_id}.annotated.parquet')
"
    """
}

process SCORE_AND_RANK {
    tag "${sample_id}"
    container 'raregeneai/core:latest'
    cpus 2
    memory '4 GB'

    input:
    tuple val(sample_id), path(annotated_parquet)
    val(hpo_terms)

    output:
    tuple val(sample_id), path("${sample_id}_ranked_genes.json")
    tuple val(sample_id), path("${sample_id}_report.html")

    script:
    """
    python3 -c "
import json
from raregeneai.config.settings import PipelineConfig
from raregeneai.pipeline.orchestrator import RareGeneAIPipeline

config = PipelineConfig()
config.ranking.top_n_genes = ${params.top_n}

pipeline = RareGeneAIPipeline(config)
hpo_list = '${hpo_terms}'.split(',')

report = pipeline.run(
    vcf_path='${annotated_parquet}',
    hpo_terms=hpo_list,
    patient_id='${sample_id}',
    output_dir='.',
)

# Save ranked genes as JSON
results = []
for gene in report.ranked_genes:
    results.append({
        'gene': gene.gene_symbol,
        'score': gene.gene_rank_score,
        'confidence': gene.confidence,
        'phenotype_score': gene.phenotype_score,
        'n_variants': gene.n_variants,
        'explanation': gene.explanation,
    })

with open('${sample_id}_ranked_genes.json', 'w') as f:
    json.dump(results, f, indent=2)
"
    """
}


// ── Workflow ──────────────────────────────────────────────────────────────────

workflow {
    // Input channel
    vcf_ch = Channel.fromPath(params.vcf)
        .map { vcf -> tuple(params.patient_id, vcf) }

    hpo_ch = Channel.value(params.hpo)

    // Pipeline stages
    NORMALIZE_VCF(vcf_ch)
    ANNOTATE_VEP(NORMALIZE_VCF.out)
    ANNOTATE_FREQUENCIES(ANNOTATE_VEP.out)
    SCORE_AND_RANK(ANNOTATE_FREQUENCIES.out, hpo_ch)

    // Collect outputs
    SCORE_AND_RANK.out[0]
        .map { sample_id, json_file -> json_file }
        .collectFile(name: 'all_results.json', storeDir: params.outdir)

    SCORE_AND_RANK.out[1]
        .map { sample_id, report -> report }
        .collectFile(storeDir: params.outdir)
}


// ── Completion handler ────────────────────────────────────────────────────────

workflow.onComplete {
    log.info """
    =======================================
    RareGeneAI Pipeline Complete
    =======================================
    Status    : ${workflow.success ? 'SUCCESS' : 'FAILED'}
    Duration  : ${workflow.duration}
    Output    : ${params.outdir}
    =======================================
    """.stripIndent()
}
