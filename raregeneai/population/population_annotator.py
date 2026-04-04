"""Population-specific allele frequency annotation and founder variant detection.

Annotates variants with local/regional population frequencies from:
  - QGP (Qatar Genome Programme)
  - GME (Greater Middle East Variome)
  - Custom local cohort databases

Detects founder variants: alleles enriched in the local population
relative to global frequencies. These are critical for Middle Eastern,
Ashkenazi, Finnish, and other founder-effect populations where globally
rare variants may be locally common and NOT causal.

Produces a population-adjusted rarity score that replaces the naive
gnomAD-only rarity when local data is available.
"""

from __future__ import annotations

import math
from pathlib import Path

from loguru import logger

from raregeneai.config.settings import PopulationConfig
from raregeneai.models.data_models import AnnotatedVariant


# ── Known Middle Eastern founder variant genes ────────────────────────────────
# These genes are known to harbor founder variants in Arab/Middle Eastern pops.
# Used as a prior when no local frequency DB is available.
ME_FOUNDER_GENES = {
    # Gene: (variant_count_in_literature, associated_disease)
    "GJB2": (5, "Hearing loss"),
    "SLC26A4": (3, "Pendred syndrome / hearing loss"),
    "MYO15A": (2, "Hearing loss"),
    "CYP1B1": (3, "Primary congenital glaucoma"),
    "TMEM216": (2, "Joubert syndrome"),
    "CC2D2A": (2, "Joubert syndrome"),
    "MCCC2": (1, "3-methylcrotonyl-CoA carboxylase deficiency"),
    "ASPM": (2, "Primary microcephaly"),
    "WDR62": (1, "Primary microcephaly"),
    "CLN6": (1, "Neuronal ceroid lipofuscinosis"),
    "CLN8": (1, "Neuronal ceroid lipofuscinosis"),
    "HEXA": (2, "Tay-Sachs disease"),
    "MEFV": (4, "Familial Mediterranean fever"),
    "HBB": (3, "Beta-thalassemia / sickle cell"),
    "G6PD": (3, "G6PD deficiency"),
    "LIPA": (1, "Wolman disease"),
    "PCCA": (1, "Propionic acidemia"),
    "SLC25A13": (1, "Citrin deficiency"),
}


class PopulationAnnotator:
    """Annotate variants with population-specific frequencies."""

    def __init__(self, config: PopulationConfig | None = None):
        self.config = config or PopulationConfig()
        self._local_db: dict[str, dict] | None = None
        self._known_founders: dict[str, dict] | None = None

    def annotate(self, variants: list[AnnotatedVariant]) -> list[AnnotatedVariant]:
        """Run population-specific frequency annotation.

        Steps:
        1. Query local population database (QGP/GME/custom)
        2. Compute population AF ratio (local/global enrichment)
        3. Detect founder variants
        4. Set population-adjusted flags
        """
        if not self.config.enabled:
            return variants

        if not variants:
            return variants

        # Load databases
        self._load_local_db()
        self._load_known_founders()

        pop_label = self.config.population or self.config.local_db_name or "local"
        logger.info(f"Annotating {len(variants)} variants with {pop_label} population frequencies")

        for var in variants:
            self._annotate_local_frequency(var)
            self._detect_founder_variant(var)

        n_local = sum(1 for v in variants if v.local_af is not None)
        n_founder = sum(1 for v in variants if v.is_founder_variant)
        n_adjusted = sum(
            1 for v in variants
            if v.local_af is not None and v.gnomad_af is not None
            and v.local_af > v.gnomad_af * 2
        )
        logger.info(
            f"Population annotation: {n_local} with local AF, "
            f"{n_founder} founder variants, {n_adjusted} with AF enrichment"
        )

        return variants

    def _load_local_db(self) -> None:
        """Load local population frequency database into memory.

        Supports:
        - QGP VCF (tabix) or TSV
        - GME Variome TSV
        - Generic TSV: chrom<TAB>pos<TAB>ref<TAB>alt<TAB>af[<TAB>ac<TAB>an<TAB>hom]
        """
        if self._local_db is not None:
            return

        self._local_db = {}

        # Try each source in priority order
        for path_attr, db_name in [
            ("qgp_af_path", "QGP"),
            ("gme_af_path", "GME"),
            ("local_af_path", self.config.local_db_name or "local"),
        ]:
            path = getattr(self.config, path_attr, None)
            if path and Path(path).exists():
                self._load_tsv_db(path, db_name)
                return

    def _load_tsv_db(self, path: str, db_name: str) -> None:
        """Load a TSV frequency database.

        Format: chrom<TAB>pos<TAB>ref<TAB>alt<TAB>af[<TAB>ac<TAB>an<TAB>hom]
        """
        count = 0
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("#") or line.startswith("chrom"):
                        continue
                    fields = line.strip().split("\t")
                    if len(fields) < 5:
                        continue

                    chrom = fields[0]
                    pos = fields[1]
                    ref = fields[2]
                    alt = fields[3]
                    af = float(fields[4])

                    key = f"{chrom}-{pos}-{ref}-{alt}"
                    entry = {"af": af, "db": db_name}

                    if len(fields) >= 6:
                        entry["ac"] = int(fields[5])
                    if len(fields) >= 7:
                        entry["an"] = int(fields[6])
                    if len(fields) >= 8:
                        entry["hom"] = int(fields[7])

                    self._local_db[key] = entry
                    count += 1

            logger.info(f"Loaded {count} variants from {db_name} database ({path})")
        except Exception as e:
            logger.error(f"Failed to load {db_name} database: {e}")

    def _load_known_founders(self) -> None:
        """Load known founder variant database."""
        if self._known_founders is not None:
            return

        self._known_founders = {}

        path = self.config.known_founders_path
        if not path or not Path(path).exists():
            return

        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    fields = line.strip().split("\t")
                    if len(fields) >= 4:
                        key = f"{fields[0]}-{fields[1]}-{fields[2]}-{fields[3]}"
                        self._known_founders[key] = {
                            "gene": fields[4] if len(fields) > 4 else "",
                            "disease": fields[5] if len(fields) > 5 else "",
                        }

            logger.info(f"Loaded {len(self._known_founders)} known founder variants")
        except Exception as e:
            logger.debug(f"Failed to load known founders: {e}")

    def _annotate_local_frequency(self, var: AnnotatedVariant) -> None:
        """Look up variant in local population database."""
        if not self._local_db:
            return

        key = var.variant.variant_key
        entry = self._local_db.get(key)

        if entry:
            var.local_af = entry["af"]
            var.local_ac = entry.get("ac", 0)
            var.local_an = entry.get("an", 0)
            var.local_hom_count = entry.get("hom", 0)
            var.local_population = entry.get("db", self.config.population)

            # Compute enrichment ratio
            gnomad = var.gnomad_af or 0.0
            if gnomad > 0:
                var.population_af_ratio = var.local_af / gnomad
            elif var.local_af > 0:
                # Present locally but absent globally = very enriched
                var.population_af_ratio = var.local_af / 1e-6  # Cap denominator
        else:
            var.local_population = self.config.population

    def _detect_founder_variant(self, var: AnnotatedVariant) -> None:
        """Detect whether this variant is a founder variant.

        A founder variant is:
        1. Enriched in the local population (local_af >> gnomad_af)
        2. Has local_af >= founder_local_af_min
        3. Has gnomad_af <= founder_global_af_max (rare globally)

        OR is in the known founder variant database.
        """
        cfg = self.config

        # Check known founder database first
        if self._known_founders:
            key = var.variant.variant_key
            if key in self._known_founders:
                var.is_founder_variant = True
                var.founder_enrichment = var.population_af_ratio or 100.0
                return

        # Check by known founder genes + population context
        if (var.gene_symbol in ME_FOUNDER_GENES
                and self.config.population in ("QGP", "GME", "saudi", "emirati", "mid", "qatari")):
            # Lower threshold for known ME founder genes
            if var.local_af is not None and var.local_af >= cfg.founder_local_af_min / 2:
                gnomad = var.gnomad_af or 0.0
                if gnomad <= cfg.founder_global_af_max * 2:
                    var.is_founder_variant = True
                    var.founder_enrichment = var.population_af_ratio or 10.0
                    return

        # General founder detection criteria
        if var.local_af is None:
            return

        gnomad = var.gnomad_af or 0.0
        local = var.local_af

        meets_local_min = local >= cfg.founder_local_af_min
        meets_global_max = gnomad <= cfg.founder_global_af_max
        meets_enrichment = (
            (var.population_af_ratio is not None and var.population_af_ratio >= cfg.founder_enrichment_threshold)
            or (gnomad == 0.0 and local >= cfg.founder_local_af_min)
        )

        if meets_local_min and meets_global_max and meets_enrichment:
            var.is_founder_variant = True
            var.founder_enrichment = var.population_af_ratio or (local / max(gnomad, 1e-6))

    def compute_population_adjusted_rarity(
        self, var: AnnotatedVariant, base_af_threshold: float = 0.01,
    ) -> float:
        """Compute population-adjusted rarity score.

        Key principle: use the MAXIMUM of local and global AF to determine
        rarity. A variant that is 5% in QGP but 0.01% in gnomAD should be
        scored as COMMON (5%) for a Qatari patient, not rare.

        Founder variants get special handling:
        - If pathogenic founder: keep high rarity (disease-relevant despite frequency)
        - If benign founder: penalize rarity (common in population = likely benign)

        Returns:
            Population-adjusted rarity score in [0, 1].
        """
        effective = var.effective_af

        # Novel in both databases
        if effective == 0.0:
            return 1.0

        # Common in any reference
        if effective > base_af_threshold:
            return 0.0

        # Founder variant handling
        if var.is_founder_variant:
            clinvar_lower = var.clinvar_significance.lower()
            if "pathogenic" in clinvar_lower:
                # Known pathogenic founder: maintain rarity score
                # (e.g. MEFV mutations in Mediterranean populations)
                return math.exp(-500 * effective)
            else:
                # Founder variant without pathogenic evidence:
                # penalize — locally common = lower prior for causality
                penalized_af = effective * self.config.local_common_penalty
                return math.exp(-2000 * penalized_af)

        # Standard exponential decay with population-adjusted AF
        return math.exp(-1000 * effective)
