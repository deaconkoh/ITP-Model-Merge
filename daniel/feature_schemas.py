"""Versioned model input schemas for canonical and inherited checkpoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    operation_dim: int
    machine_dim: int
    pair_dim: int
    operation_priority: bool
    pair_carbon: bool
    legacy: bool = False


SCHEMAS = {
    "canonical_f11_p9_v2": FeatureSchema(
        "canonical_f11_p9_v2", 11, 8, 9, operation_priority=True, pair_carbon=True
    ),
    "legacy_f11_p9_v1": FeatureSchema(
        "legacy_f11_p9_v1", 11, 8, 9, operation_priority=True, pair_carbon=True, legacy=True
    ),
    "legacy_f11_p8_v1": FeatureSchema(
        "legacy_f11_p8_v1", 11, 8, 8, operation_priority=True, pair_carbon=False, legacy=True
    ),
    "legacy_f10_p8_v1": FeatureSchema(
        "legacy_f10_p8_v1", 10, 8, 8, operation_priority=False, pair_carbon=False, legacy=True
    ),
}


def get_feature_schema(name: str) -> FeatureSchema:
    try:
        return SCHEMAS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown feature schema {name!r}; expected one of {tuple(SCHEMAS)}") from exc


def validate_config_against_schema(config) -> FeatureSchema:
    schema = get_feature_schema(config.feature_schema)
    actual = (config.fea_j_input_dim, config.fea_m_input_dim, config.fea_pair_input_dim)
    expected = (schema.operation_dim, schema.machine_dim, schema.pair_dim)
    if actual != expected:
        raise ValueError(
            f"Feature dimensions {actual} do not match schema {schema.name} {expected}. "
            "Legacy checkpoints require an explicit legacy schema."
        )
    if schema.operation_priority:
        observes_operation_slot = config.enable_priority or (schema.legacy and config.enable_carbon)
        if not observes_operation_slot:
            raise ValueError(f"Schema {schema.name} requires its eleventh operation-feature slot")
    if schema.pair_carbon and not (config.enable_carbon and config.carbon_feature):
        raise ValueError(f"Schema {schema.name} requires the carbon pair feature")
    return schema
