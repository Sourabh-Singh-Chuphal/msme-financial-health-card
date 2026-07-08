"""Unit tests for persona composition and cohort generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.synthetic_generators.persona_builder import (
    DEFAULT_COHORT_COUNTS,
    PERSONA_TYPES,
    PersonaBuilder,
    build_persona_spec,
    generate_cohort,
)
from data.synthetic_generators.utils import make_rng


def test_all_persona_types_exist() -> None:
    assert len(PERSONA_TYPES) == 5


def test_healthy_ntc_has_partial_sources(rng) -> None:
    for salt in range(30):
        spec = build_persona_spec("MSME_TEST", "healthy_ntc", make_rng(42, salt), salt)
        assert 2 <= len(spec.sources_available) <= 3
        assert "upi" in spec.sources_available or "gst" in spec.sources_available


def test_healthy_established_has_all_sources(rng) -> None:
    spec = build_persona_spec("MSME_TEST", "healthy_established", rng, 1)
    assert set(spec.sources_available) == {"gst", "upi", "aa", "epfo"}


def test_generate_one_writes_only_available_sources(persona_builder, tmp_path) -> None:
    persona_builder.output_dir = tmp_path
    generated = persona_builder.generate_one("MSME_000999", "healthy_ntc", 999)
    out = persona_builder.write_msme(generated)
    files = {p.name for p in out.iterdir()}
    assert "metadata.json" in files
    for source in generated.spec.sources_available:
        if source == "aa":
            assert "aa_daily.csv" in files
        else:
            assert f"{source}.csv" in files


def test_cohort_count_is_two_hundred(tmp_path) -> None:
    generated = generate_cohort(seed=42, output_dir=tmp_path)
    assert len(generated) == sum(DEFAULT_COHORT_COUNTS.values())
    assert (tmp_path / "manifest.json").exists()
