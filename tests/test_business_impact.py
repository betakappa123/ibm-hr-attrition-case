"""Unit tests for business scenario calculator and relative vs. percentage-point math."""

from __future__ import annotations

import numpy as np
import pytest

from workforce_retention.business_impact import (
    calculate_retention_scenario,
    convert_attrition_reduction,
)


def test_relative_reduction_vs_percentage_point_reduction_distinction() -> None:
    # 5,000 employees, 16.0% baseline attrition => 800 baseline departures
    rel_10 = convert_attrition_reduction(
        workforce_size=5000,
        baseline_attrition_rate=0.16,
        reduction_mode="relative",
        reduction_amount=0.10,
    )
    pp_10 = convert_attrition_reduction(
        workforce_size=5000,
        baseline_attrition_rate=0.16,
        reduction_mode="percentage_point",
        reduction_amount=10.0,
    )

    # 10% relative drop on 16.0% => -1.60 pp (post rate 14.4%), 80 prevented departures
    assert np.isclose(rel_10.percentage_point_reduction_pp, 1.60)
    assert np.isclose(rel_10.post_intervention_rate_pct, 14.40)
    assert np.isclose(rel_10.departures_prevented, 80.0)

    # 10.0 pp drop on 16.0% => -62.5% relative (post rate 6.0%), 500 prevented departures
    assert np.isclose(pp_10.percentage_point_reduction_pp, 10.0)
    assert np.isclose(pp_10.relative_reduction_pct, 62.5)
    assert np.isclose(pp_10.post_intervention_rate_pct, 6.0)
    assert np.isclose(pp_10.departures_prevented, 500.0)


def test_zero_intervention_efficacy_produces_zero_benefit_even_with_perfect_model() -> None:
    res = calculate_retention_scenario(
        workforce_size=2000,
        baseline_attrition_rate=0.15,
        intervention_effectiveness=0.0,
        replacement_cost_per_departure=50_000.0,
        intervention_cost_per_flagged=1_000.0,
        model_recall=1.0,
        model_fpr=0.0,
    )
    assert res.prevented_departures == 0.0
    assert res.gross_replacement_savings == 0.0
    assert res.net_cost_savings == -300_000.0


def test_breakeven_effectiveness_yields_zero_net_savings() -> None:
    probe = calculate_retention_scenario(
        workforce_size=5000,
        baseline_attrition_rate=0.1612,
        intervention_effectiveness=0.20,
        replacement_cost_per_departure=45_000.0,
        intervention_cost_per_flagged=1_500.0,
        model_recall=0.65,
        model_fpr=0.15,
    )
    eta_be = probe.breakeven_effectiveness_pct / 100.0
    at_breakeven = calculate_retention_scenario(
        workforce_size=5000,
        baseline_attrition_rate=0.1612,
        intervention_effectiveness=eta_be,
        replacement_cost_per_departure=45_000.0,
        intervention_cost_per_flagged=1_500.0,
        model_recall=0.65,
        model_fpr=0.15,
    )
    assert abs(at_breakeven.net_cost_savings) < 50.0  # Within rounding of 2-decimal %


def test_invalid_scenario_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="percentage_point"):
        convert_attrition_reduction(
            workforce_size=1000,
            baseline_attrition_rate=0.10,
            reduction_mode="percentage_point",
            reduction_amount=15.0,  # Cannot drop 15 pp from 10% baseline
        )

