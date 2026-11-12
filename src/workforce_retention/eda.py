"""Exploratory statistical analysis and publication-quality visualizations.

All statistical findings represent observational associations within the cross-sectional
IBM HR dataset and must NOT be interpreted as causal treatment effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PALETTE_MAIN = {"No": "#2b5c8f", "Yes": "#d95f02"}
COLOR_PRIMARY = "#2b5c8f"
COLOR_ACCENT = "#d95f02"
COLOR_NEUTRAL = "#6c757d"


def wilson_ci(successes: int | np.ndarray, total: int | np.ndarray, z: float = 1.959964) -> tuple[np.ndarray, np.ndarray]:
    """Compute 95% Wilson score confidence interval for binomial proportions."""
    k = np.asarray(successes, dtype=float)
    n = np.asarray(total, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, 0.0)
        denom = 1.0 + (z**2) / n
        center = (p + (z**2) / (2.0 * n)) / denom
        half_width = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
        lower = np.clip(center - half_width, 0.0, 1.0)
        upper = np.clip(center + half_width, 0.0, 1.0)
    return lower, upper


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked_p = p[order]
    q = ranked_p * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def compute_cramers_v(contingency: pd.DataFrame) -> tuple[float, float, int]:
    """Compute Chi-square statistic, p-value, dof, and bias-standard Cramér's V."""
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency, correction=False)
    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / (n * max(min_dim, 1))))
    return float(chi2), float(p_val), int(dof), cramers_v


def run_statistical_tests(
    clean_df: pd.DataFrame,
    metrics_dir: Path | None = None,
) -> pd.DataFrame:
    """Run two-sided univariate statistical tests with Benjamini-Hochberg FDR control."""
    categorical_and_ordinal = [
        "OverTime",
        "JobRole",
        "BusinessTravel",
        "MaritalStatus",
        "Department",
        "EducationField",
        "Gender",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
        "JobInvolvement",
        "StockOptionLevel",
        "JobLevel",
    ]
    continuous_vars = [
        "MonthlyIncome",
        "YearsAtCompany",
        "TotalWorkingYears",
        "YearsInCurrentRole",
        "YearsWithCurrManager",
        "Age",
        "DistanceFromHome",
        "DailyRate",
        "MonthlyRate",
        "HourlyRate",
        "PercentSalaryHike",
        "NumCompaniesWorked",
        "YearsSinceLastPromotion",
        "TrainingTimesLastYear",
    ]

    rows: list[dict[str, Any]] = []

    for col in categorical_and_ordinal:
        ct = pd.crosstab(clean_df[col], clean_df["AttritionFlag"])
        chi2, p_val, dof, cramers_v = compute_cramers_v(ct)
        rates = clean_df.groupby(col)["AttritionFlag"].mean() * 100.0
        highest_grp = str(rates.idxmax())
        lowest_grp = str(rates.idxmin())
        rows.append(
            {
                "feature": col,
                "feature_type": "categorical/ordinal",
                "test_name": f"Chi-square (dof={dof})",
                "statistic": round(chi2, 4),
                "p_value": p_val,
                "effect_size_metric": "Cramers_V",
                "effect_size": round(cramers_v, 4),
                "abs_effect_size": round(abs(cramers_v), 4),
                "direction_or_contrast": (
                    f"Highest: {highest_grp} ({rates.max():.1f}%) vs "
                    f"Lowest: {lowest_grp} ({rates.min():.1f}%)"
                ),
            }
        )

    stayers = clean_df[clean_df["AttritionFlag"] == 0]
    leavers = clean_df[clean_df["AttritionFlag"] == 1]

    for col in continuous_vars:
        x_leave = leavers[col].to_numpy(dtype=float)
        x_stay = stayers[col].to_numpy(dtype=float)
        u_stat, p_val = stats.mannwhitneyu(x_leave, x_stay, alternative="two-sided")
        # Rank-biserial correlation: positive => higher values among leavers
        r_rb = (2.0 * u_stat) / (len(x_leave) * len(x_stay)) - 1.0
        med_leave = float(np.median(x_leave))
        med_stay = float(np.median(x_stay))
        rows.append(
            {
                "feature": col,
                "feature_type": "continuous",
                "test_name": "Mann-Whitney U",
                "statistic": round(float(u_stat), 1),
                "p_value": float(p_val),
                "effect_size_metric": "Rank_Biserial_r",
                "effect_size": round(float(r_rb), 4),
                "abs_effect_size": round(abs(float(r_rb)), 4),
                "direction_or_contrast": (
                    f"Median Yes={med_leave:,.1f} vs No={med_stay:,.1f}"
                ),
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df["q_value_bh"] = benjamini_hochberg(stats_df["p_value"].to_numpy())
    stats_df["significant_fdr_05"] = stats_df["q_value_bh"] < 0.05
    stats_df = stats_df.sort_values(
        ["abs_effect_size", "p_value"], ascending=[False, True]
    ).reset_index(drop=True)

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(metrics_dir / "eda_statistical_tests.csv", index=False)

    return stats_df


def _cohort_rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, observed=False)["AttritionFlag"]
        .agg(n="count", k="sum")
        .reset_index()
    )
    grouped["rate"] = grouped["k"] / grouped["n"]
    low, high = wilson_ci(grouped["k"].to_numpy(), grouped["n"].to_numpy())
    grouped["ci_low"] = low
    grouped["ci_high"] = high
    return grouped


def generate_eda_figures(
    clean_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Generate clean, non-causal exploratory charts with 95% confidence intervals."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.02)
    base_rate = float(clean_df["AttritionFlag"].mean()) * 100.0
    saved_paths: list[Path] = []

    # -------------------------------------------------------------------------
    # Figure 1: Target Class Imbalance & Stock Option Cohort Rates
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    counts = clean_df["Attrition"].value_counts().reindex(["No", "Yes"])
    bars = axes[0].bar(
        ["Retained (No)", "Attrition (Yes)"],
        counts.to_numpy(),
        color=[PALETTE_MAIN["No"], PALETTE_MAIN["Yes"]],
        width=0.55,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, cnt in zip(bars, counts.to_numpy()):
        pct = 100.0 * cnt / len(clean_df)
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 25,
            f"n = {cnt:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(counts.to_numpy()) * 1.15)
    axes[0].set_ylabel("Employee Headcount")
    axes[0].set_title("A. Class Imbalance in Workforce Outcome (N = 1,470)", fontweight="bold")

    stock_tbl = _cohort_rate_table(clean_df, "StockOptionLevel")
    x_pos = np.arange(len(stock_tbl))
    rates_pct = stock_tbl["rate"].to_numpy() * 100.0
    yerr = np.vstack(
        [
            rates_pct - stock_tbl["ci_low"].to_numpy() * 100.0,
            stock_tbl["ci_high"].to_numpy() * 100.0 - rates_pct,
        ]
    )
    axes[1].bar(
        x_pos,
        rates_pct,
        yerr=yerr,
        capsize=5,
        color="#3b7ea1",
        width=0.55,
        edgecolor="black",
        linewidth=0.6,
        error_kw={"elinewidth": 1.4, "ecolor": "#222222"},
    )
    axes[1].axhline(base_rate, linestyle="--", color=COLOR_ACCENT, linewidth=1.5, label=f"Workforce Base Rate ({base_rate:.1f}%)")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"Level {lvl}\n(n={n})" for lvl, n in zip(stock_tbl["StockOptionLevel"], stock_tbl["n"])])
    axes[1].set_ylabel("Observed Attrition Rate (%)")
    axes[1].set_ylim(0, 36)
    axes[1].set_title("B. Attrition Rate by Stock Option Level (95% Wilson CI)", fontweight="bold")
    axes[1].legend(loc="upper right")

    fig.suptitle("Workforce Attrition Overview & Equity Grant Association", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    p1 = figures_dir / "fig01_target_and_overview.png"
    fig.savefig(p1, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p1)

    # -------------------------------------------------------------------------
    # Figure 2: OverTime & JobSatisfaction Associations
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ot_tbl = _cohort_rate_table(clean_df, "OverTime").set_index("OverTime").reindex(["No", "Yes"]).reset_index()
    ot_rates = ot_tbl["rate"].to_numpy() * 100.0
    ot_err = np.vstack(
        [
            ot_rates - ot_tbl["ci_low"].to_numpy() * 100.0,
            ot_tbl["ci_high"].to_numpy() * 100.0 - ot_rates,
        ]
    )
    ot_bars = axes[0].bar(
        [f"No OverTime\n(n={ot_tbl.loc[0, 'n']:,})", f"OverTime = Yes\n(n={ot_tbl.loc[1, 'n']:,})"],
        ot_rates,
        yerr=ot_err,
        capsize=6,
        color=[COLOR_PRIMARY, COLOR_ACCENT],
        width=0.5,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, r_val, err_up in zip(ot_bars, ot_rates, ot_err[1]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            r_val + err_up + 1.2,
            f"{r_val:.1f}%",
            ha="center",
            fontweight="bold",
        )
    axes[0].axhline(base_rate, linestyle="--", color="#333333", linewidth=1.3, label=f"Base Rate ({base_rate:.1f}%)")
    axes[0].set_ylim(0, 42)
    axes[0].set_ylabel("Observed Attrition Rate (%)")
    axes[0].set_title("A. Attrition Rate by OverTime Status (95% CI)", fontweight="bold")
    axes[0].legend(loc="upper left")

    js_tbl = _cohort_rate_table(clean_df, "JobSatisfaction")
    js_labels = {1: "1: Low", 2: "2: Medium", 3: "3: High", 4: "4: Very High"}
    js_rates = js_tbl["rate"].to_numpy() * 100.0
    js_err = np.vstack(
        [
            js_rates - js_tbl["ci_low"].to_numpy() * 100.0,
            js_tbl["ci_high"].to_numpy() * 100.0 - js_rates,
        ]
    )
    js_x = np.arange(len(js_tbl))
    js_bars = axes[1].bar(
        js_x,
        js_rates,
        yerr=js_err,
        capsize=5,
        color="#4a89aa",
        width=0.55,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, r_val, err_up in zip(js_bars, js_rates, js_err[1]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            r_val + err_up + 1.0,
            f"{r_val:.1f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[1].axhline(base_rate, linestyle="--", color=COLOR_ACCENT, linewidth=1.3, label=f"Base Rate ({base_rate:.1f}%)")
    axes[1].set_xticks(js_x)
    axes[1].set_xticklabels([f"{js_labels[val]}\n(n={n})" for val, n in zip(js_tbl["JobSatisfaction"], js_tbl["n"])])
    axes[1].set_ylim(0, 35)
    axes[1].set_ylabel("Observed Attrition Rate (%)")
    axes[1].set_title("B. Attrition Rate by Job Satisfaction (95% CI)", fontweight="bold")
    axes[1].legend(loc="upper right")

    fig.text(
        0.5,
        -0.03,
        "Note: Observational cross-sectional associations; error bars denote 95% Wilson confidence intervals.",
        ha="center",
        fontsize=9.5,
        style="italic",
        color="#444444",
    )
    fig.tight_layout()
    p2 = figures_dir / "fig02_overtime_and_satisfaction.png"
    fig.savefig(p2, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p2)

    # -------------------------------------------------------------------------
    # Figure 3: Attrition by JobRole with Headcount & Median Income Context
    # -------------------------------------------------------------------------
    role_tbl = _cohort_rate_table(clean_df, "JobRole")
    med_inc = clean_df.groupby("JobRole")["MonthlyIncome"].median().rename("median_income")
    role_tbl = role_tbl.merge(med_inc, on="JobRole").sort_values("rate", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    y_pos = np.arange(len(role_tbl))
    r_pct = role_tbl["rate"].to_numpy() * 100.0
    r_low = role_tbl["ci_low"].to_numpy() * 100.0
    r_high = role_tbl["ci_high"].to_numpy() * 100.0

    colors = [COLOR_ACCENT if r > base_rate else COLOR_PRIMARY for r in r_pct]
    ax.barh(y_pos, r_pct, color=colors, height=0.6, alpha=0.88, edgecolor="black", linewidth=0.5)
    ax.errorbar(
        r_pct,
        y_pos,
        xerr=np.vstack([r_pct - r_low, r_high - r_pct]),
        fmt="none",
        ecolor="#1a1a1a",
        elinewidth=1.4,
        capsize=4,
    )
    ax.axvline(base_rate, linestyle="--", color="#222222", linewidth=1.4, label=f"Workforce Base Rate ({base_rate:.1f}%)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [
            f"{row.JobRole} (n={row.n}, med ${row.median_income:,.0f})"
            for row in role_tbl.itertuples()
        ]
    )
    for idx, row in enumerate(role_tbl.itertuples()):
        ax.text(
            r_high[idx] + 0.8,
            idx,
            f"{row.rate * 100.0:.1f}% ({row.k}/{row.n})",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlim(0, 58)
    ax.set_xlabel("Observed Attrition Rate (%) with 95% Wilson CI")
    ax.set_title("Observed Attrition Rate by Job Role (Annotated with Headcount & Median Monthly Income)", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p3 = figures_dir / "fig03_job_role_attrition.png"
    fig.savefig(p3, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p3)

    # -------------------------------------------------------------------------
    # Figure 4: Compensation, Company Tenure, and OverTime x Income Interaction
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    sns.boxplot(
        data=clean_df,
        x="Attrition",
        y="MonthlyIncome",
        order=["No", "Yes"],
        hue="Attrition",
        palette=PALETTE_MAIN,
        legend=False,
        width=0.48,
        ax=axes[0],
    )
    axes[0].set_title("A. Monthly Income by Attrition", fontweight="bold")
    axes[0].set_xlabel("Attrition Outcome")
    axes[0].set_ylabel("Monthly Income (USD)")

    tenure_bins = [-1, 1, 4, 9, 100]
    tenure_labels = ["0-1 yr", "2-4 yrs", "5-9 yrs", "10+ yrs"]
    temp_df = clean_df.assign(
        TenureBand=pd.cut(clean_df["YearsAtCompany"], bins=tenure_bins, labels=tenure_labels)
    )
    ten_tbl = _cohort_rate_table(temp_df, "TenureBand")
    ten_rates = ten_tbl["rate"].to_numpy() * 100.0
    ten_err = np.vstack(
        [
            ten_rates - ten_tbl["ci_low"].to_numpy() * 100.0,
            ten_tbl["ci_high"].to_numpy() * 100.0 - ten_rates,
        ]
    )
    ten_x = np.arange(len(ten_tbl))
    axes[1].bar(
        ten_x,
        ten_rates,
        yerr=ten_err,
        capsize=5,
        color="#3b7ea1",
        width=0.55,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].axhline(base_rate, linestyle="--", color=COLOR_ACCENT, linewidth=1.3, label=f"Base Rate ({base_rate:.1f}%)")
    axes[1].set_xticks(ten_x)
    axes[1].set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(ten_tbl["TenureBand"], ten_tbl["n"])])
    axes[1].set_ylim(0, 46)
    axes[1].set_ylabel("Observed Attrition Rate (%)")
    axes[1].set_title("B. Attrition by Company Tenure Band", fontweight="bold")
    axes[1].legend(loc="upper right")

    inc_q = pd.qcut(
        clean_df["MonthlyIncome"],
        q=4,
        labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"],
    )
    inter_df = (
        clean_df.assign(IncomeQuartile=inc_q)
        .groupby(["IncomeQuartile", "OverTime"], observed=False)["AttritionFlag"]
        .mean()
        .mul(100.0)
        .reset_index()
    )
    sns.barplot(
        data=inter_df,
        x="IncomeQuartile",
        y="AttritionFlag",
        hue="OverTime",
        hue_order=["No", "Yes"],
        palette={"No": COLOR_PRIMARY, "Yes": COLOR_ACCENT},
        edgecolor="black",
        linewidth=0.5,
        ax=axes[2],
    )
    axes[2].set_title("C. Income Quartile × OverTime Rate", fontweight="bold")
    axes[2].set_xlabel("Monthly Income Quartile")
    axes[2].set_ylabel("Observed Attrition Rate (%)")
    axes[2].set_ylim(0, 60)
    axes[2].legend(title="OverTime", loc="upper right")

    fig.suptitle("Compensation, Tenure & Workload Associations with Employee Attrition", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    p4 = figures_dir / "fig04_compensation_and_tenure.png"
    fig.savefig(p4, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p4)

    # -------------------------------------------------------------------------
    # Figure 5: Top 15 Univariate Effect Sizes with FDR Significance
    # -------------------------------------------------------------------------
    top_stats = stats_df.head(15).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y_idx = np.arange(len(top_stats))
    bar_colors = [
        COLOR_ACCENT if row.significant_fdr_05 else COLOR_NEUTRAL
        for row in top_stats.itertuples()
    ]
    ax.barh(
        y_idx,
        top_stats["abs_effect_size"].to_numpy(),
        color=bar_colors,
        height=0.62,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(y_idx)
    ax.set_yticklabels(
        [
            f"{row.feature} ({'V' if row.effect_size_metric == 'Cramers_V' else '|r_rb|'})"
            for row in top_stats.itertuples()
        ]
    )
    for idx, row in enumerate(top_stats.itertuples()):
        sig_tag = f"q={row.q_value_bh:.2e}" if row.q_value_bh < 0.001 else f"q={row.q_value_bh:.3f}"
        ax.text(
            row.abs_effect_size + 0.006,
            idx,
            f"{row.abs_effect_size:.3f} ({sig_tag})",
            va="center",
            fontsize=8.8,
        )
    ax.set_xlim(0, float(top_stats["abs_effect_size"].max()) * 1.28)
    ax.set_xlabel("Univariate Effect Size Magnitude (Cramér's V or |Rank-Biserial r|)")
    ax.set_title("Top 15 Univariate Associations with Attrition (Benjamini-Hochberg FDR Adjusted)", fontweight="bold")
    fig.tight_layout()
    p5 = figures_dir / "fig05_statistical_effect_sizes.png"
    fig.savefig(p5, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p5)

    return saved_paths

