"""Synthetic alternate-data generators for MSME credit assessment development.

Banks reject New-to-Credit MSMEs for lack of traditional documents; this package
fabricates realistic GST, UPI, Account Aggregator, and EPFO records so downstream
ingestion, feature, and scoring layers can be built and validated without real PII.
"""

from data.synthetic_generators.persona_builder import (
    PERSONA_TYPES,
    PersonaBuilder,
    generate_cohort,
)

__all__ = [
    "PERSONA_TYPES",
    "PersonaBuilder",
    "generate_cohort",
]
