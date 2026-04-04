"""PED file parser for family/trio analysis."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from raregeneai.models.data_models import Pedigree, PedigreeMember


class PedigreeParser:
    """Parse standard PED format pedigree files."""

    SEX_MAP = {"1": "male", "2": "female", "0": "unknown"}
    AFFECTED_MAP = {"2": True, "1": False, "0": False, "-9": False}

    def parse(self, ped_path: str | Path) -> list[Pedigree]:
        """Parse PED file into Pedigree objects.

        PED format (tab-separated):
        family_id  individual_id  father_id  mother_id  sex  affected_status

        Returns one Pedigree per family.
        """
        ped_path = Path(ped_path)
        families: dict[str, list[PedigreeMember]] = {}

        with open(ped_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) < 6:
                    fields = line.split()
                if len(fields) < 6:
                    logger.warning(f"Skipping malformed PED line: {line}")
                    continue

                fam_id, ind_id, father, mother, sex, affected = fields[:6]

                member = PedigreeMember(
                    individual_id=ind_id,
                    family_id=fam_id,
                    father_id=father if father != "0" else None,
                    mother_id=mother if mother != "0" else None,
                    sex=self.SEX_MAP.get(sex, "unknown"),
                    affected=self.AFFECTED_MAP.get(affected, False),
                    vcf_sample_id=ind_id,
                )

                families.setdefault(fam_id, []).append(member)

        pedigrees = []
        for fam_id, members in families.items():
            # Proband = first affected individual
            proband_id = ""
            for m in members:
                if m.affected:
                    proband_id = m.individual_id
                    break

            pedigrees.append(Pedigree(
                family_id=fam_id,
                members=members,
                proband_id=proband_id,
            ))

        logger.info(f"Parsed {len(pedigrees)} families from PED file")
        return pedigrees
