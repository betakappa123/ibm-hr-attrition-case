"""Configurable retention scenario calculator and cost-benefit sensitivity analysis.

IMPORTANT CAUSAL DISCLAIMER:
Predictive performance (PR-AUC, ROC-AUC, Recall, Precision) only measures how accurately
a model identifies employees at elevated risk of leaving under historical conditions.
It does NOT establish whether an HR intervention causally reduces attrition. All outputs
from this module are hypothetical scenario projections conditioned on user-assumed
intervention effectiveness (eta), not realized financial outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class AttritionReductionComparison:
    """Explicit conversion between relative (%) and percentage-point (pp) attrition reduction."""

    baseline_rate_pct: float
    post_intervention_rate_pct: float
    relative_reduction_pct: float
    percentage_point_reduction_pp: float
    departures_baseline: float
    departures_prevented: float
    departures_post: float


@dataclass(frozen=True)
class ScenarioResult:
    """Structured output of a configurable workforce retention scenario."""

    scenario_name: str
    workforce_size: int
    baseline_attrition_rate_pct: float
    assumed_intervention_effectiveness_pct: float
    model_recall: float
    model_fpr: float
    replacement_cost_per_departure: float
    intervention_cost_per_flagged: float
    baseline_departures: float
    flagged_employees: float
    true_leavers_contacted: float
    false_positives_contacted: float
    prevented_departures: float
    post_attrition_rate_pct: float
    achieved_relative_reduction_pct: float
    achieved_percentage_point_reduction_pp: float
    gross_replacement_savings: float
    total_intervention_cost: float
    net_cost_savings: float
    roi_pct: float
    breakeven_effectiveness_pct: float


def convert_attrition_reduction(
    *,
    workforce_size: int,
    baseline_attrition_rate: float,
    reduction_mode: str,
    reduction_amount: float,
) -> AttritionReductionComparison:
    """Convert between relative reduction and percentage-point (pp) reduction.

    Parameters
    ----------
    workforce_size : int
        Total active employee headcount (N > 0).
    baseline_attrition_rate : float
        Baseline annual attrition proportion in (0, 1), e.g., 0.1612 for 16.12%.
    reduction_mode : {'relative', 'percentage_point'}
        - 'relative': `reduction_amount` is a relative proportion in [0, 1]
          (e.g., 0.10 means a 10% relative drop: 16.12% -> 14.51%, i.e., -1.61 pp).
        - 'percentage_point': `reduction_amount` is in percentage points [0, 100*r0]
          (e.g., 2.0 means a 2.0 pp drop: 16.12% -> 14.12%, i.e., -12.41% relative).
    reduction_amount : float
        Magnitude of reduction interpreted according to `reduction_mode`.
    """
    if workforce_size <= 0:
        raise ValueError("workforce_size must be positive.")
    if not (0.0 < baseline_attrition_rate < 1.0):
        raise ValueError("baseline_attrition_rate must be strictly between 0 and 1.")

    base_pct = baseline_attrition_rate * 100.0
    if reduction_mode == "relative":
        if not (0.0 <= reduction_amount <= 1.0):
            raise ValueError("For 'relative' mode, reduction_amount must be in [0.0, 1.0].")
        rel_pct = reduction_amount * 100.0
        pp_drop = base_pct * reduction_amount
        post_pct = base_pct - pp_drop
    elif reduction_mode == "percentage_point":
        if not (0.0 <= reduction_amount <= base_pct):
            raise ValueError(
                f"For 'percentage_point' mode, reduction_amount must be in [0.0, {base_pct:.4f}]."
            )
        pp_drop = reduction_amount
        post_pct = base_pct - pp_drop
        rel_pct = (pp_drop / base_pct) * 100.0
    else:
        raise ValueError("reduction_mode must be 'relative' or 'percentage_point'.")

    dep_base = workforce_size * baseline_attrition_rate
    dep_post = workforce_size * (post_pct / 100.0)
    dep_prev = dep_base - dep_post

    return AttritionReductionComparison(
        baseline_rate_pct=round(base_pct, 4),
        post_intervention_rate_pct=round(post_pct, 4),
        relative_reduction_pct=round(rel_pct, 4),
        percentage_point_reduction_pp=round(pp_drop, 4),
        departures_baseline=round(dep_base, 2),
        departures_prevented=round(dep_prev, 2),
        departures_post=round(dep_post, 2),
    )


def calculate_retention_scenario(
    *,
    workforce_size: int,
    baseline_attrition_rate: float,
    intervention_effectiveness: float,
    replacement_cost_per_departure: float,
    intervention_cost_per_flagged: float,
    model_recall: float = 1.0,
    model_fpr: float = 1.0,
    scenario_name: str = "Custom Scenario",
) -> ScenarioResult:
    """Evaluate hypothetical retention benefits and net savings under explicit assumptions."""
    if workforce_size <= 0:
        raise ValueError("workforce_size must be > 0.")
    if not (0.0 < baseline_attrition_rate < 1.0):
        raise ValueError("baseline_attrition_rate must be in (0, 1).")
    if not (0.0 <= intervention_effectiveness <= 1.0):
        raise ValueError("intervention_effectiveness must be in [0, 1].")
    if replacement_cost_per_departure < 0.0:
        raise ValueError("replacement_cost_per_departure must be >= 0.")
    if intervention_cost_per_flagged < 0.0:
        raise ValueError("intervention_cost_per_flagged must be >= 0.")
    if not (0.0 <= model_recall <= 1.0) or not (0.0 <= model_fpr <= 1.0):
        raise ValueError("model_recall and model_fpr must be in [0, 1].")

    baseline_departures = workforce_size * baseline_attrition_rate
    baseline_stayers = workforce_size * (1.0 - baseline_attrition_rate)

    tp_contacted = baseline_departures * model_recall
    fp_contacted = baseline_stayers * model_fpr
    flagged_total = tp_contacted + fp_contacted

    prevented = tp_contacted * intervention_effectiveness
    post_departures = baseline_departures - prevented
    post_rate = post_departures / workforce_size

    achieved_rel_pct = (prevented / baseline_departures) * 100.0 if baseline_departures > 0 else 0.0
    achieved_pp = (baseline_attrition_rate - post_rate) * 100.0

    gross_savings = prevented * replacement_cost_per_departure
    program_cost = flagged_total * intervention_cost_per_flagged
    net_savings = gross_savings - program_cost
    roi_pct = (net_savings / program_cost) * 100.0 if program_cost > 0 else 0.0

    if tp_contacted > 0 and replacement_cost_per_departure > 0:
        breakeven_eff = (
            flagged_total * intervention_cost_per_flagged
        ) / (tp_contacted * replacement_cost_per_departure)
        breakeven_eff_pct = round(breakeven_eff * 100.0, 4)
    else:
        breakeven_eff_pct = float("inf")

    return ScenarioResult(
        scenario_name=scenario_name,
        workforce_size=int(workforce_size),
        baseline_attrition_rate_pct=round(baseline_attrition_rate * 100.0, 3),
        assumed_intervention_effectiveness_pct=round(intervention_effectiveness * 100.0, 2),
        model_recall=round(model_recall, 4),
        model_fpr=round(model_fpr, 4),
        replacement_cost_per_departure=round(replacement_cost_per_departure, 2),
        intervention_cost_per_flagged=round(intervention_cost_per_flagged, 2),
        baseline_departures=round(baseline_departures, 2),
        flagged_employees=round(flagged_total, 2),
        true_leavers_contacted=round(tp_contacted, 2),
        false_positives_contacted=round(fp_contacted, 2),
        prevented_departures=round(prevented, 2),
        post_attrition_rate_pct=round(post_rate * 100.0, 3),
        achieved_relative_reduction_pct=round(achieved_rel_pct, 3),
        achieved_percentage_point_reduction_pp=round(achieved_pp, 3),
        gross_replacement_savings=round(gross_savings, 2),
        total_intervention_cost=round(program_cost, 2),
        net_cost_savings=round(net_savings, 2),
        roi_pct=round(roi_pct, 2),
        breakeven_effectiveness_pct=breakeven_eff_pct,
    )


def generate_scenario_table_and_figure(
    *,
    workforce_size: int,
    baseline_attrition_rate: float,
    default_effectiveness: float,
    replacement_cost: float,
    intervention_cost: float,
    model_recall: float,
    model_fpr: float,
    metrics_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """Generate benchmark scenarios table and sensitivity visualization."""
    scenarios = [
        calculate_retention_scenario(
            scenario_name="Conservative ML-Targeted (eta=10%)",
            workforce_size=workforce_size,
            baseline_attrition_rate=baseline_attrition_rate,
            intervention_effectiveness=0.10,
            replacement_cost_per_departure=replacement_cost,
            intervention_cost_per_flagged=intervention_cost,
            model_recall=model_recall,
            model_fpr=model_fpr,
        ),
        calculate_retention_scenario(
            scenario_name=f"Moderate ML-Targeted (eta={int(default_effectiveness * 100)}%)",
            workforce_size=workforce_size,
            baseline_attrition_rate=baseline_attrition_rate,
            intervention_effectiveness=default_effectiveness,
            replacement_cost_per_departure=replacement_cost,
            intervention_cost_per_flagged=intervention_cost,
            model_recall=model_recall,
            model_fpr=model_fpr,
        ),
        calculate_retention_scenario(
            scenario_name="Optimistic ML-Targeted (eta=25%)",
            workforce_size=workforce_size,
            baseline_attrition_rate=baseline_attrition_rate,
            intervention_effectiveness=0.25,
            replacement_cost_per_departure=replacement_cost,
            intervention_cost_per_flagged=intervention_cost,
            model_recall=model_recall,
            model_fpr=model_fpr,
        ),
        calculate_retention_scenario(
            scenario_name=f"Untargeted Blanket Program (eta={int(default_effectiveness * 100)}%)",
            workforce_size=workforce_size,
            baseline_attrition_rate=baseline_attrition_rate,
            intervention_effectiveness=default_effectiveness,
            replacement_cost_per_departure=replacement_cost,
            intervention_cost_per_flagged=intervention_cost,
            model_recall=1.0,
            model_fpr=1.0,
        ),
        calculate_retention_scenario(
            scenario_name="Zero-Efficacy Null Intervention (eta=0%)",
            workforce_size=workforce_size,
            baseline_attrition_rate=baseline_attrition_rate,
            intervention_effectiveness=0.00,
            replacement_cost_per_departure=replacement_cost,
            intervention_cost_per_flagged=intervention_cost,
            model_recall=model_recall,
            model_fpr=model_fpr,
        ),
    ]
    scen_df = pd.DataFrame([asdict(s) for s in scenarios])

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        scen_df.to_csv(metrics_dir / "business_scenarios.csv", index=False)

    fig_path: Path | None = None
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
        fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))

        # Panel A: Net Savings Heatmap over Intervention Effectiveness x Replacement Cost
        eff_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        rep_grid = [25000, 35000, 45000, 60000, 75000]
        heat_matrix = np.zeros((len(rep_grid), len(eff_grid)))
        for i, r_cost in enumerate(rep_grid):
            for j, eff in enumerate(eff_grid):
                res = calculate_retention_scenario(
                    workforce_size=workforce_size,
                    baseline_attrition_rate=baseline_attrition_rate,
                    intervention_effectiveness=eff,
                    replacement_cost_per_departure=float(r_cost),
                    intervention_cost_per_flagged=intervention_cost,
                    model_recall=model_recall,
                    model_fpr=model_fpr,
                )
                heat_matrix[i, j] = res.net_cost_savings / 1000.0  # in $K

        sns.heatmap(
            heat_matrix,
            annot=True,
            fmt="+.0f",
            cmap="RdYlGn",
            center=0.0,
            xticklabels=[f"{int(e * 100)}%" for e in eff_grid],
            yticklabels=[f"${r // 1000}k" for r in rep_grid],
            cbar_kws={"label": "Projected Net Savings (USD Thousands)"},
            ax=axes[0],
        )
        axes[0].set_xlabel("Assumed Causal Intervention Effectiveness (eta)")
        axes[0].set_ylabel("Replacement Cost per Departure")
        axes[0].set_title(
            f"A. Hypothetical Net Savings (USD K) | N={workforce_size:,}, Outreach={intervention_cost:,.0f} USD",
            fontweight="bold",
            fontsize=10.5,
        )

        # Panel B: Explicit Contrast Between Relative (%) vs Percentage-Point (pp) Reduction
        labels_cmp = ["5% Relative\n(-0.81 pp)", "10% Relative\n(-1.61 pp)", "15% Relative\n(-2.42 pp)", "5.0 pp Drop\n(-31.0% Rel)"]
        modes_cmp = [
            convert_attrition_reduction(
                workforce_size=workforce_size,
                baseline_attrition_rate=baseline_attrition_rate,
                reduction_mode="relative",
                reduction_amount=0.05,
            ),
            convert_attrition_reduction(
                workforce_size=workforce_size,
                baseline_attrition_rate=baseline_attrition_rate,
                reduction_mode="relative",
                reduction_amount=0.10,
            ),
            convert_attrition_reduction(
                workforce_size=workforce_size,
                baseline_attrition_rate=baseline_attrition_rate,
                reduction_mode="relative",
                reduction_amount=0.15,
            ),
            convert_attrition_reduction(
                workforce_size=workforce_size,
                baseline_attrition_rate=baseline_attrition_rate,
                reduction_mode="percentage_point",
                reduction_amount=5.0,
            ),
        ]
        prevented_vals = [m.departures_prevented for m in modes_cmp]
        bar_cols = ["#2b5c8f", "#2b5c8f", "#2b5c8f", "#d95f02"]
        bars = axes[1].bar(labels_cmp, prevented_vals, color=bar_cols, width=0.52, edgecolor="black", linewidth=0.6)
        for bar, m_obj in zip(bars, modes_cmp):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 5,
                f"{m_obj.departures_prevented:.0f} retained\n(post: {m_obj.post_intervention_rate_pct:.1f}%)",
                ha="center",
                fontsize=8.8,
                fontweight="bold",
            )
        axes[1].set_ylim(0, max(prevented_vals) * 1.24)
        axes[1].set_ylabel(f"Prevented Annual Departures (Workforce N = {workforce_size:,})")
        axes[1].set_title("B. Relative (%) vs. Percentage-Point (pp) Reduction Scale", fontweight="bold", fontsize=10.5)

        fig.text(
            0.5,
            -0.03,
            "Illustrative scenario projections only: model accuracy does not prove causal intervention efficacy.",
            ha="center",
            fontsize=9.5,
            style="italic",
            color="#444444",
        )
        fig.tight_layout()
        fig_path = figures_dir / "fig10_business_scenario_sensitivity.png"
        fig.savefig(fig_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    return scen_df, fig_path
