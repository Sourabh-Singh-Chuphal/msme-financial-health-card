"""Shared pytest fixtures for synthetic MSME generator tests."""

from __future__ import annotations

import numpy as np
import pytest

from data.synthetic_generators.persona_builder import PersonaBuilder
from data.synthetic_generators.utils import GenerationWindow, make_rng


@pytest.fixture
def rng() -> np.random.Generator:
    return make_rng(42, salt=0)


@pytest.fixture
def window() -> GenerationWindow:
    return GenerationWindow.last_n_months(12, anchor=__import__("datetime").date(2026, 7, 1))


@pytest.fixture
def persona_builder() -> PersonaBuilder:
    return PersonaBuilder(seed=42, output_dir="data/raw_test")
