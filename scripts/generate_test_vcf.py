"""Generate a test VCF file for RareGeneAI development and testing.

Creates a synthetic VCF with known pathogenic variants for benchmarking.
"""

from __future__ import annotations

import gzip
import random
from pathlib import Path


def generate_test_vcf(output_path: str = "test_data/test_sample.vcf.gz"):
    """Generate a synthetic VCF with test variants."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Known disease gene variants for testing
    test_variants = [
        # Gene: SCN1A (Dravet syndrome - epilepsy)
        ("chr2", 166049061, "rs121918823", "G", "A", "SCN1A", "stop_gained"),
        ("chr2", 166056301, ".", "C", "T", "SCN1A", "missense_variant"),
        # Gene: BRCA1 (breast cancer)
        ("chr17", 43094464, "rs80357906", "G", "A", "BRCA1", "frameshift_variant"),
        # Gene: CFTR (cystic fibrosis)
        ("chr7", 117559590, "rs75961395", "ATCT", "A", "CFTR", "frameshift_variant"),
        # Gene: MECP2 (Rett syndrome)
        ("chrX", 154031611, "rs28934907", "G", "A", "MECP2", "missense_variant"),
        # Background variants (benign/common)
        ("chr1", 1000000, ".", "A", "G", "NONE", "intergenic"),
        ("chr1", 2000000, ".", "T", "C", "NONE", "synonymous"),
        ("chr3", 50000000, ".", "G", "T", "GENE1", "missense_variant"),
        ("chr5", 100000000, ".", "C", "A", "GENE2", "intron_variant"),
        ("chr10", 75000000, ".", "A", "T", "GENE3", "missense_variant"),
    ]

    # Add more random background variants
    chroms = [f"chr{i}" for i in range(1, 23)]
    for _ in range(90):
        chrom = random.choice(chroms)
        pos = random.randint(1000000, 200000000)
        ref = random.choice("ACGT")
        alt = random.choice([b for b in "ACGT" if b != ref])
        test_variants.append((chrom, pos, ".", ref, alt, "BGENE", "intron_variant"))

    header = """##fileformat=VCFv4.2
##source=RareGeneAI_TestGenerator
##reference=GRCh38
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr2,length=242193529>
##contig=<ID=chr3,length=198295559>
##contig=<ID=chr5,length=181538259>
##contig=<ID=chr7,length=159345973>
##contig=<ID=chr10,length=133797422>
##contig=<ID=chr17,length=83257441>
##contig=<ID=chrX,length=156040895>
##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">
##INFO=<ID=CSQ,Number=1,Type=String,Description="Consequence">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE_001"""

    lines = [header]

    # Sort variants by chrom, pos
    test_variants.sort(key=lambda x: (x[0], x[1]))

    for chrom, pos, vid, ref, alt, gene, csq in test_variants:
        dp = random.randint(20, 100)
        gq = random.randint(30, 99)

        # Make disease variants heterozygous, background mostly ref
        if gene in ("SCN1A", "BRCA1", "CFTR", "MECP2"):
            gt = "0/1"
            ad_ref = dp // 2
            ad_alt = dp - ad_ref
        else:
            gt = random.choice(["0/0", "0/0", "0/0", "0/1"])
            if gt == "0/1":
                ad_ref = dp // 2
                ad_alt = dp - ad_ref
            else:
                ad_ref = dp
                ad_alt = 0

        info = f"GENE={gene};CSQ={csq}"
        sample = f"{gt}:{dp}:{gq}:{ad_ref},{ad_alt}"

        lines.append(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t{gq}\tPASS\t{info}\tGT:DP:GQ:AD\t{sample}")

    content = "\n".join(lines) + "\n"

    if output_path.endswith(".gz"):
        with gzip.open(output_path, "wt") as f:
            f.write(content)
    else:
        with open(output_path, "w") as f:
            f.write(content)

    print(f"Generated test VCF: {output_path}")
    print(f"  Total variants: {len(test_variants)}")
    print(f"  Disease variants: SCN1A(2), BRCA1(1), CFTR(1), MECP2(1)")
    print(f"  Background variants: {len(test_variants) - 5}")

    return output_path


if __name__ == "__main__":
    generate_test_vcf()
