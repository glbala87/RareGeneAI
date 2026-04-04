#!/bin/bash
# Download reference data for RareGeneAI
# =======================================
set -euo pipefail

DATA_DIR="${1:-data/reference}"
mkdir -p "$DATA_DIR"

echo "=== RareGeneAI Reference Data Download ==="
echo "Target directory: $DATA_DIR"

# ── HPO Ontology ────────────────────────────────────────────────────────────
echo "[1/5] Downloading HPO ontology..."
wget -q -O "$DATA_DIR/hp.obo" \
    "https://purl.obolibrary.org/obo/hp.obo" || \
    curl -sL "https://purl.obolibrary.org/obo/hp.obo" -o "$DATA_DIR/hp.obo"
echo "  Done: hp.obo"

# ── Gene-Phenotype Associations ────────────────────────────────────────────
echo "[2/5] Downloading gene-phenotype associations..."
wget -q -O "$DATA_DIR/genes_to_phenotype.txt" \
    "https://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt" || \
    curl -sL "https://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt" \
    -o "$DATA_DIR/genes_to_phenotype.txt"
echo "  Done: genes_to_phenotype.txt"

# ── ClinVar VCF ────────────────────────────────────────────────────────────
echo "[3/5] Downloading ClinVar VCF (GRCh38)..."
wget -q -O "$DATA_DIR/clinvar.vcf.gz" \
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz" || \
    echo "  Warning: ClinVar download failed (optional)"
if [ -f "$DATA_DIR/clinvar.vcf.gz" ]; then
    wget -q -O "$DATA_DIR/clinvar.vcf.gz.tbi" \
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi" || true
fi
echo "  Done: clinvar.vcf.gz"

# ── VEP Cache ──────────────────────────────────────────────────────────────
echo "[4/5] VEP cache setup..."
echo "  For VEP cache, run: vep_install --AUTO cf --ASSEMBLY GRCh38 --SPECIES homo_sapiens"
echo "  Or download from: https://ftp.ensembl.org/pub/release-112/variation/vep/"
mkdir -p "$DATA_DIR/vep_cache"

# ── CADD (optional) ───────────────────────────────────────────────────────
echo "[5/5] CADD scores (optional)..."
echo "  Download from: https://cadd.gs.washington.edu/download"
echo "  Place files at:"
echo "    $DATA_DIR/whole_genome_SNVs.tsv.gz"
echo "    $DATA_DIR/gnomad.genomes.r4.0.indel.tsv.gz"

echo ""
echo "=== Reference data download complete ==="
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR/"
