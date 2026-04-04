"""Unit tests for core data models."""

import pytest

from raregeneai.models.data_models import (
    AnnotatedVariant,
    FunctionalImpact,
    GeneCandidate,
    HPOTerm,
    InheritanceMode,
    PatientPhenotype,
    Pedigree,
    PedigreeMember,
    Variant,
    Zygosity,
)


class TestVariant:
    def test_variant_key(self):
        v = Variant(chrom="chr1", pos=12345, ref="A", alt="G")
        assert v.variant_key == "chr1-12345-A-G"

    def test_is_snv(self):
        v = Variant(chrom="1", pos=100, ref="A", alt="T")
        assert v.is_snv
        assert not v.is_indel

    def test_is_indel(self):
        v = Variant(chrom="1", pos=100, ref="AT", alt="A")
        assert v.is_indel
        assert not v.is_snv

    def test_default_zygosity(self):
        v = Variant(chrom="1", pos=100, ref="A", alt="G")
        assert v.zygosity == Zygosity.UNKNOWN


class TestAnnotatedVariant:
    def test_is_rare_novel(self):
        v = Variant(chrom="1", pos=100, ref="A", alt="G")
        av = AnnotatedVariant(variant=v, gnomad_af=None)
        assert av.is_rare
        assert av.is_novel

    def test_is_rare_low_af(self):
        v = Variant(chrom="1", pos=100, ref="A", alt="G")
        av = AnnotatedVariant(variant=v, gnomad_af=0.001)
        assert av.is_rare
        assert not av.is_novel

    def test_not_rare_high_af(self):
        v = Variant(chrom="1", pos=100, ref="A", alt="G")
        av = AnnotatedVariant(variant=v, gnomad_af=0.05)
        assert not av.is_rare


class TestGeneCandidate:
    def test_n_variants(self):
        v1 = AnnotatedVariant(variant=Variant(chrom="1", pos=100, ref="A", alt="G"))
        v2 = AnnotatedVariant(variant=Variant(chrom="1", pos=200, ref="C", alt="T"))
        gc = GeneCandidate(gene_symbol="BRCA1", variants=[v1, v2])
        assert gc.n_variants == 2

    def test_has_lof(self):
        v = AnnotatedVariant(
            variant=Variant(chrom="1", pos=100, ref="A", alt="G"),
            impact=FunctionalImpact.HIGH,
        )
        gc = GeneCandidate(gene_symbol="TP53", variants=[v])
        assert gc.has_lof_variant

    def test_no_lof(self):
        v = AnnotatedVariant(
            variant=Variant(chrom="1", pos=100, ref="A", alt="G"),
            impact=FunctionalImpact.MODERATE,
        )
        gc = GeneCandidate(gene_symbol="TP53", variants=[v])
        assert not gc.has_lof_variant


class TestPatientPhenotype:
    def test_creation(self):
        terms = [HPOTerm(id="HP:0001250", name="Seizures")]
        pp = PatientPhenotype(patient_id="P001", hpo_terms=terms)
        assert pp.patient_id == "P001"
        assert len(pp.hpo_terms) == 1

    def test_negated_terms(self):
        pp = PatientPhenotype(
            patient_id="P001",
            hpo_terms=[HPOTerm(id="HP:0001250")],
            negated_terms=[HPOTerm(id="HP:0000252")],
        )
        assert len(pp.negated_terms) == 1


class TestPedigree:
    def test_trio(self):
        members = [
            PedigreeMember(individual_id="child", family_id="F1", affected=True),
            PedigreeMember(individual_id="father", family_id="F1"),
            PedigreeMember(individual_id="mother", family_id="F1"),
        ]
        ped = Pedigree(family_id="F1", members=members, proband_id="child")
        assert ped.is_trio
        assert ped.proband.individual_id == "child"
        assert len(ped.affected_members) == 1
