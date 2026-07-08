"""Compose synthetic MSME personas from alternate-data generators.

Credit models fail in demos when every synthetic borrower looks identical. This
module defines five archetypes — including the hackathon-critical healthy_ntc
(thin-file but strong alternate signals) — and samples realistic variance within
each archetype so downstream scoring can distinguish healthy from risky profiles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.synthetic_generators.aa_generator import AAProfile, generate_aa_statements, summarize_aa
from data.synthetic_generators.epfo_generator import EPFOProfile, generate_epfo_records, summarize_epfo
from data.synthetic_generators.gst_generator import GSTProfile, generate_gst_records, summarize_gst
from data.synthetic_generators.upi_generator import UPIProfile, generate_upi_logs, summarize_upi
from data.synthetic_generators.utils import BusinessType, GenerationWindow, make_rng

PERSONA_TYPES: tuple[str, ...] = (
    "healthy_established",
    "healthy_ntc",
    "risky_declining",
    "risky_volatile",
    "seasonal_business",
)

ALL_SOURCES: tuple[str, ...] = ("gst", "upi", "aa", "epfo")

# Cohort size targets (sum = 200).
DEFAULT_COHORT_COUNTS: dict[str, int] = {
    "healthy_established": 40,
    "healthy_ntc": 50,
    "risky_declining": 35,
    "risky_volatile": 35,
    "seasonal_business": 40,
}

# Typical thin-file source combinations for New-to-Credit MSMEs.
NTC_SOURCE_COMBOS: tuple[tuple[str, ...], ...] = (
    ("gst", "upi"),
    ("upi", "aa"),
    ("gst", "upi", "aa"),
    ("upi", "epfo"),
    ("gst", "upi", "epfo"),
    ("gst", "aa"),
    ("gst", "upi"),
)


@dataclass
class MSMEPersonaSpec:
    """Fully-resolved persona instance ready for data generation."""

    msme_id: str
    persona_type: str
    business_type: BusinessType
    sources_available: list[str]
    gst_profile: GSTProfile | None = None
    upi_profile: UPIProfile | None = None
    aa_profile: AAProfile | None = None
    epfo_profile: EPFOProfile | None = None
    rng_salt: int = 0


@dataclass
class GeneratedMSME:
    """In-memory bundle of generated records for one synthetic MSME."""

    spec: MSMEPersonaSpec
    gst: pd.DataFrame | None = None
    upi: pd.DataFrame | None = None
    aa_daily: pd.DataFrame | None = None
    aa_monthly: pd.DataFrame | None = None
    epfo: pd.DataFrame | None = None

    def summary_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "msme_id": self.spec.msme_id,
            "persona_type": self.spec.persona_type,
            "business_type": self.spec.business_type.value,
            "sources_available": self.spec.sources_available,
        }
        if self.gst is not None:
            stats["gst"] = summarize_gst(self.gst)
        if self.upi is not None:
            stats["upi"] = summarize_upi(self.upi)
        if self.aa_daily is not None:
            stats["aa"] = summarize_aa(self.aa_daily)
        if self.epfo is not None:
            stats["epfo"] = summarize_epfo(self.epfo)
        return stats


def _sample_business_type(rng: np.random.Generator, persona: str) -> BusinessType:
    if persona == "seasonal_business":
        weights = [0.70, 0.20, 0.10]
    elif persona == "healthy_ntc":
        weights = [0.55, 0.30, 0.15]
    else:
        weights = [0.35, 0.40, 0.25]
    idx = int(rng.choice(len(BusinessType), p=weights))
    return list(BusinessType)[idx]


def _jitter(rng: np.random.Generator, base: float, pct: float) -> float:
    return float(base * rng.uniform(1 - pct, 1 + pct))


def _build_gst_profile(persona: str, rng: np.random.Generator) -> GSTProfile:
    turnover = _jitter(rng, 1_200_000, 0.35)

    if persona == "healthy_established":
        return GSTProfile(
            base_monthly_turnover=turnover,
            monthly_turnover_growth=rng.uniform(0.004, 0.012),
            filing_delay_mean_days=rng.uniform(-1, 1),
            filing_delay_std_days=rng.uniform(0.5, 1.5),
            late_filing_probability=rng.uniform(0.0, 0.04),
            mismatch_probability=rng.uniform(0.0, 0.02),
            seasonality_strength=rng.uniform(0.15, 0.30),
        )
    if persona == "healthy_ntc":
        return GSTProfile(
            base_monthly_turnover=_jitter(rng, 650_000, 0.40),
            monthly_turnover_growth=rng.uniform(0.006, 0.015),
            filing_delay_mean_days=rng.uniform(-1, 2),
            late_filing_probability=rng.uniform(0.02, 0.08),
            mismatch_probability=rng.uniform(0.01, 0.04),
            seasonality_strength=rng.uniform(0.20, 0.40),
        )
    if persona == "risky_declining":
        return GSTProfile(
            base_monthly_turnover=turnover,
            monthly_turnover_growth=rng.uniform(-0.025, -0.008),
            filing_delay_mean_days=rng.uniform(4, 12),
            filing_delay_std_days=rng.uniform(2, 6),
            late_filing_probability=rng.uniform(0.35, 0.65),
            mismatch_probability=rng.uniform(0.08, 0.20),
            seasonality_strength=rng.uniform(0.10, 0.25),
        )
    if persona == "risky_volatile":
        return GSTProfile(
            base_monthly_turnover=turnover,
            monthly_turnover_growth=rng.uniform(-0.005, 0.008),
            turnover_noise_std=rng.uniform(0.12, 0.22),
            turnover_volatility=rng.uniform(0.18, 0.35),
            filing_delay_mean_days=rng.uniform(0, 8),
            filing_delay_std_days=rng.uniform(3, 10),
            late_filing_probability=rng.uniform(0.15, 0.45),
            mismatch_probability=rng.uniform(0.05, 0.18),
            seasonality_strength=rng.uniform(0.05, 0.20),
        )
    # seasonal_business
    return GSTProfile(
        base_monthly_turnover=_jitter(rng, 900_000, 0.30),
        monthly_turnover_growth=rng.uniform(0.002, 0.010),
        filing_delay_mean_days=rng.uniform(-1, 2),
        late_filing_probability=rng.uniform(0.02, 0.10),
        mismatch_probability=rng.uniform(0.01, 0.05),
        seasonality_strength=rng.uniform(0.55, 0.85),
    )


def _build_upi_profile(persona: str, rng: np.random.Generator) -> UPIProfile:
    if persona == "healthy_established":
        return UPIProfile(
            inflow_scale=_jitter(rng, 1.1, 0.25),
            monthly_inflow_growth=rng.uniform(0.004, 0.010),
            refund_rate_delta=rng.uniform(-0.005, 0.005),
            seasonality_strength=rng.uniform(0.20, 0.35),
        )
    if persona == "healthy_ntc":
        return UPIProfile(
            inflow_scale=_jitter(rng, 1.0, 0.30),
            monthly_inflow_growth=rng.uniform(0.005, 0.014),
            refund_rate_delta=rng.uniform(-0.003, 0.008),
            seasonality_strength=rng.uniform(0.25, 0.45),
            missing_day_probability=rng.uniform(0.01, 0.04),
        )
    if persona == "risky_declining":
        return UPIProfile(
            inflow_scale=_jitter(rng, 0.95, 0.20),
            monthly_inflow_growth=rng.uniform(-0.020, -0.006),
            refund_rate_delta=rng.uniform(0.015, 0.035),
            seasonality_strength=rng.uniform(0.15, 0.30),
        )
    if persona == "risky_volatile":
        return UPIProfile(
            inflow_scale=_jitter(rng, 1.0, 0.35),
            monthly_inflow_growth=rng.uniform(-0.008, 0.006),
            inflow_volatility=rng.uniform(0.25, 0.45),
            refund_rate_delta=rng.uniform(0.010, 0.040),
            txn_count_scale=_jitter(rng, 1.0, 0.40),
            seasonality_strength=rng.uniform(0.10, 0.25),
            missing_day_probability=rng.uniform(0.04, 0.12),
        )
    return UPIProfile(
        inflow_scale=_jitter(rng, 1.05, 0.30),
        monthly_inflow_growth=rng.uniform(0.003, 0.009),
        seasonality_strength=rng.uniform(0.60, 0.90),
        refund_rate_delta=rng.uniform(-0.002, 0.010),
    )


def _build_aa_profile(persona: str, rng: np.random.Generator) -> AAProfile:
    if persona == "healthy_established":
        return AAProfile(
            base_opening_balance=_jitter(rng, 450_000, 0.35),
            daily_net_inflow_mean=_jitter(rng, 12_000, 0.30),
            bounce_base_rate=rng.uniform(0.0, 0.015),
            overdraft_usage_probability=rng.uniform(0.0, 0.03),
            balance_volatility=rng.uniform(0.04, 0.08),
        )
    if persona == "healthy_ntc":
        return AAProfile(
            base_opening_balance=_jitter(rng, 180_000, 0.40),
            daily_net_inflow_mean=_jitter(rng, 9_000, 0.35),
            bounce_base_rate=rng.uniform(0.005, 0.025),
            overdraft_usage_probability=rng.uniform(0.02, 0.08),
            balance_volatility=rng.uniform(0.06, 0.12),
        )
    if persona == "risky_declining":
        return AAProfile(
            base_opening_balance=_jitter(rng, 320_000, 0.25),
            monthly_balance_growth=rng.uniform(-0.012, -0.003),
            daily_net_inflow_mean=_jitter(rng, 4_000, 0.35),
            bounce_base_rate=rng.uniform(0.06, 0.14),
            bounce_trend=rng.uniform(0.008, 0.025),
            overdraft_usage_probability=rng.uniform(0.12, 0.30),
            balance_volatility=rng.uniform(0.10, 0.18),
        )
    if persona == "risky_volatile":
        return AAProfile(
            base_opening_balance=_jitter(rng, 280_000, 0.45),
            daily_net_inflow_mean=_jitter(rng, 6_000, 0.50),
            daily_net_inflow_std=_jitter(rng, 28_000, 0.40),
            bounce_base_rate=rng.uniform(0.04, 0.12),
            overdraft_usage_probability=rng.uniform(0.08, 0.25),
            balance_volatility=rng.uniform(0.20, 0.35),
        )
    return AAProfile(
        base_opening_balance=_jitter(rng, 380_000, 0.30),
        daily_net_inflow_mean=_jitter(rng, 10_000, 0.35),
        bounce_base_rate=rng.uniform(0.005, 0.025),
        overdraft_usage_probability=rng.uniform(0.03, 0.10),
        balance_volatility=rng.uniform(0.08, 0.15),
    )


def _build_epfo_profile(persona: str, rng: np.random.Generator) -> EPFOProfile:
    headcount = int(rng.integers(8, 45))
    if persona == "healthy_established":
        return EPFOProfile(
            base_employee_count=headcount,
            monthly_headcount_growth=rng.uniform(0.003, 0.010),
            pf_regularity_rate=rng.uniform(0.94, 0.99),
            base_churn_rate=rng.uniform(0.015, 0.04),
        )
    if persona == "healthy_ntc":
        return EPFOProfile(
            base_employee_count=int(rng.integers(5, 25)),
            monthly_headcount_growth=rng.uniform(0.004, 0.012),
            pf_regularity_rate=rng.uniform(0.88, 0.97),
            base_churn_rate=rng.uniform(0.02, 0.06),
        )
    if persona == "risky_declining":
        return EPFOProfile(
            base_employee_count=headcount,
            monthly_headcount_growth=rng.uniform(-0.015, -0.004),
            pf_regularity_rate=rng.uniform(0.55, 0.82),
            base_churn_rate=rng.uniform(0.08, 0.18),
            churn_trend=rng.uniform(0.005, 0.015),
        )
    if persona == "risky_volatile":
        return EPFOProfile(
            base_employee_count=headcount,
            monthly_headcount_growth=rng.uniform(-0.005, 0.006),
            headcount_noise_std=rng.uniform(0.08, 0.18),
            pf_regularity_rate=rng.uniform(0.65, 0.88),
            base_churn_rate=rng.uniform(0.06, 0.16),
        )
    return EPFOProfile(
        base_employee_count=int(rng.integers(10, 35)),
        monthly_headcount_growth=rng.uniform(0.001, 0.008),
        pf_regularity_rate=rng.uniform(0.90, 0.98),
        base_churn_rate=rng.uniform(0.04, 0.10),
        seasonality_strength=rng.uniform(0.35, 0.55),
    )


def _resolve_sources(persona: str, rng: np.random.Generator) -> list[str]:
    if persona == "healthy_ntc":
        combo = NTC_SOURCE_COMBOS[int(rng.integers(0, len(NTC_SOURCE_COMBOS)))]
        return list(combo)
    return list(ALL_SOURCES)


def build_persona_spec(
    msme_id: str,
    persona_type: str,
    rng: np.random.Generator,
    rng_salt: int,
) -> MSMEPersonaSpec:
    """Sample a concrete MSME instance from a persona archetype."""
    if persona_type not in PERSONA_TYPES:
        raise ValueError(f"Unknown persona_type: {persona_type}")

    business_type = _sample_business_type(rng, persona_type)
    sources = _resolve_sources(persona_type, rng)

    spec = MSMEPersonaSpec(
        msme_id=msme_id,
        persona_type=persona_type,
        business_type=business_type,
        sources_available=sources,
        rng_salt=rng_salt,
    )

    if "gst" in sources:
        spec.gst_profile = _build_gst_profile(persona_type, rng)
    if "upi" in sources:
        spec.upi_profile = _build_upi_profile(persona_type, rng)
    if "aa" in sources:
        spec.aa_profile = _build_aa_profile(persona_type, rng)
    if "epfo" in sources:
        spec.epfo_profile = _build_epfo_profile(persona_type, rng)

    return spec


class PersonaBuilder:
    """Generate and persist synthetic MSME datasets."""

    def __init__(self, seed: int = 42, output_dir: Path | str | None = None) -> None:
        self.seed = seed
        self.output_dir = Path(output_dir or "data/raw")
        self.window = GenerationWindow.last_n_months(12)

    def generate_one(self, msme_id: str, persona_type: str, rng_salt: int) -> GeneratedMSME:
        rng = make_rng(self.seed, rng_salt)
        spec = build_persona_spec(msme_id, persona_type, rng, rng_salt)
        return self._generate_from_spec(spec, rng)

    def _generate_from_spec(self, spec: MSMEPersonaSpec, rng: np.random.Generator) -> GeneratedMSME:
        result = GeneratedMSME(spec=spec)

        if spec.gst_profile is not None:
            result.gst = generate_gst_records(self.window, spec.gst_profile, rng)
        if spec.upi_profile is not None:
            result.upi = generate_upi_logs(self.window, spec.business_type, spec.upi_profile, rng)
        if spec.aa_profile is not None:
            result.aa_daily = generate_aa_statements(self.window, spec.aa_profile, rng)
            result.aa_monthly = result.aa_daily.attrs.get("monthly_summary")
        if spec.epfo_profile is not None:
            result.epfo = generate_epfo_records(self.window, spec.epfo_profile, rng)

        return result

    def write_msme(self, generated: GeneratedMSME) -> Path:
        """Persist one MSME to data/raw/{msme_id}/."""
        out = self.output_dir / generated.spec.msme_id
        out.mkdir(parents=True, exist_ok=True)

        metadata = {
            "msme_id": generated.spec.msme_id,
            "persona_type": generated.spec.persona_type,
            "business_type": generated.spec.business_type.value,
            "sources_available": generated.spec.sources_available,
            "generation_seed": self.seed,
            "rng_salt": generated.spec.rng_salt,
            "window_start": self.window.start.isoformat(),
            "window_end": self.window.end.isoformat(),
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if generated.gst is not None:
            generated.gst.to_csv(out / "gst.csv", index=False)
        if generated.upi is not None:
            generated.upi.to_csv(out / "upi.csv", index=False)
        if generated.aa_daily is not None:
            generated.aa_daily.to_csv(out / "aa_daily.csv", index=False)
        if generated.aa_monthly is not None:
            generated.aa_monthly.to_csv(out / "aa_monthly.csv", index=False)
        if generated.epfo is not None:
            generated.epfo.to_csv(out / "epfo.csv", index=False)

        return out

    def generate_and_write(self, msme_id: str, persona_type: str, rng_salt: int) -> GeneratedMSME:
        generated = self.generate_one(msme_id, persona_type, rng_salt)
        self.write_msme(generated)
        return generated


def generate_cohort(
    counts: dict[str, int] | None = None,
    seed: int = 42,
    output_dir: Path | str | None = None,
) -> list[GeneratedMSME]:
    """Generate the full synthetic MSME cohort and write to data/raw/."""
    counts = counts or DEFAULT_COHORT_COUNTS
    builder = PersonaBuilder(seed=seed, output_dir=output_dir)
    generated: list[GeneratedMSME] = []
    salt = 0

    for persona_type, n in counts.items():
        for _ in range(n):
            salt += 1
            msme_id = f"MSME_{salt:06d}"
            generated.append(builder.generate_and_write(msme_id, persona_type, salt))

    manifest = {
        "seed": seed,
        "total_msme_count": len(generated),
        "persona_counts": counts,
        "msme_ids": [g.spec.msme_id for g in generated],
    }
    manifest_path = Path(output_dir or "data/raw") / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return generated
