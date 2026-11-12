"""Command-line interface for Workforce Retention pipeline and scenario calculator."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from workforce_retention.business_impact import (
    calculate_retention_scenario,
    convert_attrition_reduction,
    generate_scenario_table_and_figure,
)
from workforce_retention.config import ProjectConfig
from workforce_retention.data_prep import (
    DatasetNotFoundError,
    load_clean_dataset,
    prepare_feature_matrix,
    run_sql_data_preparation,
)
from workforce_retention.eda import generate_eda_figures, run_statistical_tests
from workforce_retention.fairness import (
    evaluate_subgroup_performance,
    run_demographic_ablation,
)
from workforce_retention.modeling import (
    compute_classification_metrics,
    generate_modeling_figures,
    train_and_evaluate_models,
)


def cmd_prepare_data(cfg: ProjectConfig) -> None:
    res = run_sql_data_preparation(
        raw_csv_path=cfg.raw_csv_path,
        sqlite_db_path=cfg.sqlite_db_path,
        clean_csv_path=cfg.clean_csv_path,
        sql_dir=cfg.sql_dir,
        metrics_dir=cfg.metrics_dir,
    )
    rep = res["validation_report"]
    print(
        f"[prepare-data] Validated {rep['raw_row_count']:,} raw rows -> "
        f"{rep['clean_row_count']:,} clean rows ({rep['clean_column_count']} columns). "
        f"Dropped constant columns: {rep['constant_columns_dropped']}."
    )


def cmd_eda(cfg: ProjectConfig) -> None:
    clean_df = load_clean_dataset(cfg.clean_csv_path)
    stats_df = run_statistical_tests(clean_df, metrics_dir=cfg.metrics_dir)
    figs = generate_eda_figures(clean_df, stats_df, figures_dir=cfg.figures_dir)
    sig_cnt = int(stats_df["significant_fdr_05"].sum())
    print(
        f"[eda] Completed {len(stats_df)} statistical tests ({sig_cnt} FDR-significant at q<0.05). "
        f"Saved {len(figs)} figures to '{cfg.figures_dir.relative_to(cfg.raw_csv_path.parents[2])}'."
    )


def cmd_train(cfg: ProjectConfig) -> None:
    clean_df = load_clean_dataset(cfg.clean_csv_path)
    X, y = prepare_feature_matrix(clean_df, drop_sensitive=False)

    mod_res = train_and_evaluate_models(
        X,
        y,
        test_size=cfg.test_size,
        cv_folds=cfg.cv_folds,
        random_seed=cfg.random_seed,
        threshold_criterion=cfg.threshold_selection_metric,
        permutation_repeats=cfg.permutation_repeats,
        metrics_dir=cfg.metrics_dir,
        models_dir=cfg.models_dir,
    )
    generate_modeling_figures(mod_res, figures_dir=cfg.figures_dir)

    primary_name = mod_res["primary_model_name"]
    primary_thr = mod_res["primary_threshold"]
    y_test = mod_res["y_test"]
    primary_probs = mod_res["test_probabilities"][primary_name]

    evaluate_subgroup_performance(
        clean_df=clean_df,
        test_indices=mod_res["X_test"].index,
        y_test=y_test,
        y_prob=primary_probs,
        threshold=primary_thr,
        metrics_dir=cfg.metrics_dir,
        figures_dir=cfg.figures_dir,
    )
    run_demographic_ablation(
        clean_df,
        test_size=cfg.test_size,
        cv_folds=cfg.cv_folds,
        random_seed=cfg.random_seed,
        metrics_dir=cfg.metrics_dir,
    )

    test_m = compute_classification_metrics(y_test.to_numpy(), primary_probs, primary_thr)
    generate_scenario_table_and_figure(
        workforce_size=cfg.default_workforce_size,
        baseline_attrition_rate=cfg.default_baseline_attrition,
        default_effectiveness=cfg.default_intervention_effectiveness,
        replacement_cost=cfg.default_replacement_cost,
        intervention_cost=cfg.default_intervention_cost,
        model_recall=float(test_m["recall"]),
        model_fpr=float(test_m["fpr"]),
        metrics_dir=cfg.metrics_dir,
        figures_dir=cfg.figures_dir,
    )

    print(
        f"[train] Primary model selected by CV PR-AUC: '{primary_name}' "
        f"(Val threshold t*={primary_thr:.2f} | Test PR-AUC={test_m['pr_auc']:.3f}, "
        f"ROC-AUC={test_m['roc_auc']:.3f}, F1={test_m['f1']:.3f})."
    )


def cmd_scenario(args: argparse.Namespace, cfg: ProjectConfig) -> None:
    res = calculate_retention_scenario(
        scenario_name="CLI Custom Scenario",
        workforce_size=args.workforce_size,
        baseline_attrition_rate=args.baseline_attrition,
        intervention_effectiveness=args.intervention_effectiveness,
        replacement_cost_per_departure=args.replacement_cost,
        intervention_cost_per_flagged=args.intervention_cost,
        model_recall=args.model_recall,
        model_fpr=args.model_fpr,
    )
    rel_demo = convert_attrition_reduction(
        workforce_size=args.workforce_size,
        baseline_attrition_rate=args.baseline_attrition,
        reduction_mode="relative",
        reduction_amount=0.10,
    )
    pp_demo = convert_attrition_reduction(
        workforce_size=args.workforce_size,
        baseline_attrition_rate=args.baseline_attrition,
        reduction_mode="percentage_point",
        reduction_amount=2.0,
    )
    payload = {
        "disclaimer": (
            "Hypothetical scenario projection only. Predictive accuracy does not prove "
            "that an intervention causally reduces employee attrition."
        ),
        "scenario_result": asdict(res),
        "terminology_reference": {
            "ten_percent_relative_reduction": asdict(rel_demo),
            "two_percentage_point_reduction": asdict(pp_demo),
        },
    }
    print(json.dumps(payload, indent=2))


def build_parser(cfg: ProjectConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workforce-retention",
        description="Reproducible SQL, EDA, ML, and Business Impact pipeline for Employee Attrition.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="Ingest raw CSV into SQLite, validate, and export cleaned data.")
    sub.add_parser("eda", help="Run statistical tests and generate exploratory figures.")
    sub.add_parser("train", help="Train/tune 4 classifiers, select validation threshold, audit subgroups.")
    sub.add_parser("run-all", help="Execute prepare-data, eda, train, and scenario generation end-to-end.")

    scen = sub.add_parser("scenario", help="Run configurable business retention scenario calculator.")
    scen.add_argument("--workforce-size", type=int, default=cfg.default_workforce_size)
    scen.add_argument("--baseline-attrition", type=float, default=cfg.default_baseline_attrition)
    scen.add_argument("--intervention-effectiveness", type=float, default=cfg.default_intervention_effectiveness)
    scen.add_argument("--replacement-cost", type=float, default=cfg.default_replacement_cost)
    scen.add_argument("--intervention-cost", type=float, default=cfg.default_intervention_cost)
    scen.add_argument("--model-recall", type=float, default=0.62)
    scen.add_argument("--model-fpr", type=float, default=0.14)

    return parser


def main(argv: list[str] | None = None) -> int:
    cfg = ProjectConfig.load()
    parser = build_parser(cfg)
    args = parser.parse_args(argv)

    try:
        if args.command == "prepare-data":
            cmd_prepare_data(cfg)
        elif args.command == "eda":
            cmd_eda(cfg)
        elif args.command == "train":
            cmd_train(cfg)
        elif args.command == "run-all":
            cmd_prepare_data(cfg)
            cmd_eda(cfg)
            cmd_train(cfg)
        elif args.command == "scenario":
            cmd_scenario(args, cfg)
    except DatasetNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

