"""Inspect early-termination and stalling from full-run JSON/NPZ artifacts.

Usage examples
--------------
python tests/full_run_analysis.py --seed 4321
python tests/full_run_analysis.py --json /abs/path/to/run.json
python tests/full_run_analysis.py --seed 4321 --outer-iters 15,46,56 --solver fast --component 2
python tests/full_run_analysis.py --seed 4321 --prompt-outer-iters --solver fast --component 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.plotter import DykstraPlotter
from utils.projection_result import ProjectionResult


def _default_results_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "results" / "full_experiment_benchmarks" / "full_run_iterates"


def _resolve_json_path(
    json_path: str | None,
    seed: int | None,
    results_dir: Path,
) -> Path:
    if json_path is not None:
        path = Path(json_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"JSON path does not exist: {path}")
        return path

    if seed is None:
        raise ValueError("Provide either --json or --seed.")

    matches = sorted(
        results_dir.glob(f"*SEED={seed}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No JSON artifacts found for seed={seed} in {results_dir}."
        )
    return matches[0]


def _resolve_npz_path(payload: dict[str, Any], json_path: Path) -> Path:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("JSON payload missing metadata object.")

    npz_pointer = metadata.get("npz_pointer", {})
    if not isinstance(npz_pointer, dict):
        raise ValueError("JSON metadata missing npz_pointer object.")

    absolute_path = npz_pointer.get("absolute_path")
    if isinstance(absolute_path, str) and absolute_path.strip():
        return Path(absolute_path).resolve()

    rel_path = npz_pointer.get("relative_to_json_dir")
    if isinstance(rel_path, str) and rel_path.strip():
        return (json_path.parent / rel_path).resolve()

    raise ValueError("Could not resolve NPZ path from JSON metadata.")


def _array_head_tail(arr: np.ndarray, n: int = 8) -> tuple[str, str]:
    head = np.array2string(arr[:n], precision=6, separator=", ")
    tail = np.array2string(arr[-n:], precision=6, separator=", ")
    return head, tail


def _parse_outer_iter_list(raw: str | None) -> list[int]:
    if raw is None:
        return []
    cleaned = raw.strip()
    if not cleaned:
        return []
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    parsed: list[int] = []
    for token in cleaned.split(","):
        tok = token.strip()
        if not tok:
            continue
        parsed.append(int(tok))
    return parsed


def _print_component_solver_summary(
    component_dim: int,
    solver_label: str,
    solver_meta: dict[str, Any],
    npz_data: Any,
) -> dict[str, Any]:
    key_iters = solver_meta.get("projection_iterations_run_key")
    key_early = solver_meta.get("projection_terminated_early_key")
    key_reason = solver_meta.get("projection_termination_reason_key")
    proj_full_keys = solver_meta.get("projection_results_full_keys")

    if not isinstance(key_iters, str) or key_iters not in npz_data:
        print(
            f"  [{solver_label}] missing '{key_iters}' in NPZ; "
            "termination summary unavailable."
        )
        return {}

    iters = np.asarray(npz_data[key_iters], dtype=int)
    total_outer = int(iters.size)

    early: np.ndarray | None = None
    if isinstance(key_early, str) and key_early in npz_data:
        early = np.asarray(npz_data[key_early], dtype=bool)
    else:
        early = np.zeros_like(iters, dtype=bool)

    reasons: np.ndarray | None = None
    if isinstance(key_reason, str) and key_reason in npz_data:
        reasons = np.asarray(npz_data[key_reason], dtype=object)
    else:
        reasons = np.asarray(["unknown"] * total_outer, dtype=object)

    early_count = int(np.sum(early))
    zero_iter_early = int(np.sum(early & (iters == 0)))
    positive_iter_early = int(np.sum(early & (iters > 0)))
    full_budget_count = int(np.sum(~early))

    print(f"  [{solver_label}] total_outer={total_outer}")
    print(
        "    early_terminated="
        f"{early_count}/{total_outer} ({(100.0 * early_count / max(total_outer, 1)):.2f}%)"
    )
    print(f"    early_at_zero_iters={zero_iter_early}/{total_outer}")
    print(f"    early_after_positive_iters={positive_iter_early}/{total_outer}")
    print(f"    full_budget_fallback={full_budget_count}/{total_outer}")
    print(
        "    iterations_run_stats="
        f"(min={int(np.min(iters))}, max={int(np.max(iters))}, "
        f"mean={float(np.mean(iters)):.3f})"
    )

    reason_counts = Counter(map(str, reasons.tolist()))
    print(f"    termination_reason_breakdown={dict(reason_counts)}")

    stalled_outer: np.ndarray = np.array([], dtype=int)
    if isinstance(proj_full_keys, dict):
        stalled_key = proj_full_keys.get("stalled_errors")
        if isinstance(stalled_key, str) and stalled_key in npz_data:
            stalled_series = np.asarray(npz_data[stalled_key], dtype=object)
            flagged: list[int] = []
            for outer_idx, series in enumerate(stalled_series):
                if series is None:
                    continue
                arr = np.asarray(series, dtype=float)
                if arr.size > 0 and np.any(~np.isnan(arr)):
                    flagged.append(int(outer_idx))
            stalled_outer = np.asarray(flagged, dtype=int)
    if stalled_outer.size > 0:
        print(
            "    stalling_detected=True "
            f"(outer_iters={stalled_outer.tolist()})"
        )
    else:
        print("    stalling_detected=False")

    return {
        "iters": iters,
        "early": early,
        "reasons": reasons,
        "stalled_outer": stalled_outer,
    }


def _print_error_samples(
    solver_label: str,
    solver_meta: dict[str, Any],
    summary: dict[str, Any],
    npz_data: Any,
    sample_full_errors: int,
    sample_early_errors: int,
) -> None:
    if not summary:
        return

    proj_full_keys = solver_meta.get("projection_results_full_keys")
    if not isinstance(proj_full_keys, dict):
        print(f"    [{solver_label}] no projection_results_full_keys; skipping error samples.")
        return

    sq_key = proj_full_keys.get("squared_errors")
    if not isinstance(sq_key, str) or sq_key not in npz_data:
        print(f"    [{solver_label}] no squared_errors key in projection_results_full; skipping.")
        return

    iters = np.asarray(summary["iters"], dtype=int)
    early = np.asarray(summary["early"], dtype=bool)
    sq_all = np.asarray(npz_data[sq_key], dtype=object)

    outer_full = np.where(~early)[0]
    outer_early_pos = np.where(early & (iters > 0))[0]

    def _print_for_indices(
        tag: str,
        outer_indices: np.ndarray,
        n_samples: int,
        n_head_tail: int,
    ) -> None:
        if n_samples <= 0:
            return
        picked = outer_indices[:n_samples]
        if picked.size == 0:
            print(f"    {tag}: none")
            return
        print(f"    {tag} sample_outer_indices={picked.tolist()}")
        for outer_idx in picked:
            err_obj = sq_all[int(outer_idx)]
            if err_obj is None:
                print(f"      outer={int(outer_idx)} squared_errors=None")
                continue
            arr = np.asarray(err_obj, dtype=float)
            head, tail = _array_head_tail(arr, n=n_head_tail)
            print(
                f"      outer={int(outer_idx)} len={len(arr)} "
                f"start={arr[0]:.6e} end={arr[-1]:.6e}"
            )
            print(f"        head{n_head_tail}={head}")
            print(f"        tail{n_head_tail}={tail}")

    _print_for_indices(
        tag="full_budget_error_samples",
        outer_indices=outer_full,
        n_samples=sample_full_errors,
        n_head_tail=3,
    )
    _print_for_indices(
        tag="early_positive_iter_error_samples",
        outer_indices=outer_early_pos,
        n_samples=sample_early_errors,
        n_head_tail=12,
    )


def _build_projection_result_for_outer(
    npz_data: Any,
    solver_meta: dict[str, Any],
    outer_idx: int,
) -> ProjectionResult | None:
    proj_full_keys = solver_meta.get("projection_results_full_keys")
    if not isinstance(proj_full_keys, dict):
        return None

    sq_key = proj_full_keys.get("squared_errors")
    st_key = proj_full_keys.get("stalled_errors")
    cv_key = proj_full_keys.get("converged_errors")
    if not isinstance(sq_key, str) or sq_key not in npz_data:
        return None

    sq_obj = np.asarray(npz_data[sq_key], dtype=object)[outer_idx]
    if sq_obj is None:
        return None
    sq = np.asarray(sq_obj, dtype=float)

    st: np.ndarray | None = None
    cv: np.ndarray | None = None
    if isinstance(st_key, str) and st_key in npz_data:
        st_obj = np.asarray(npz_data[st_key], dtype=object)[outer_idx]
        st = None if st_obj is None else np.asarray(st_obj, dtype=float)
    if isinstance(cv_key, str) and cv_key in npz_data:
        cv_obj = np.asarray(npz_data[cv_key], dtype=object)[outer_idx]
        cv = None if cv_obj is None else np.asarray(cv_obj, dtype=float)

    if st is None:
        st = np.full_like(sq, np.nan, dtype=float)
    if cv is None:
        cv = np.full_like(sq, np.nan, dtype=float)

    return ProjectionResult(
        projection=np.array([], dtype=float),
        squared_errors=sq,
        stalled_errors=st,
        converged_errors=cv,
    )


def _plot_selected_outer_iterations(
    output_dir: Path,
    json_path: Path,
    component_dim: int,
    solver_label: str,
    selected_outer: list[int],
    solver_meta: dict[str, Any],
    npz_data: Any,
) -> None:
    if not selected_outer:
        return

    key_iters = solver_meta.get("projection_iterations_run_key")
    if not isinstance(key_iters, str) or key_iters not in npz_data:
        print(
            f"    [{solver_label}] cannot plot selected outer iters: "
            "missing projection_iterations_run key."
        )
        return

    total_outer = int(np.asarray(npz_data[key_iters]).size)
    valid = [idx for idx in selected_outer if 0 <= idx < total_outer]
    invalid = [idx for idx in selected_outer if idx < 0 or idx >= total_outer]
    if invalid:
        print(
            f"    [{solver_label}] skipped out-of-range outer iters: {invalid} "
            f"(valid range: 0..{max(total_outer - 1, 0)})"
        )
    if not valid:
        print(f"    [{solver_label}] no valid selected outer iters to plot.")
        return

    results: list[ProjectionResult] = []
    labels: list[str] = []
    for outer_idx in valid:
        result = _build_projection_result_for_outer(
            npz_data=npz_data,
            solver_meta=solver_meta,
            outer_idx=outer_idx,
        )
        if result is None:
            continue
        results.append(result)
        labels.append(f"Outer {outer_idx} - {solver_label}")

    if not results:
        print(f"    [{solver_label}] selected outer iters have no track_error data.")
        return

    max_iter = max(len(r.squared_errors) - 1 for r in results if r.squared_errors is not None)
    stem = json_path.stem
    filename = (
        f"{stem}_comp{component_dim}_{solver_label}_selected_outer_"
        + "-".join(str(v) for v in valid)
        + ".png"
    )

    plotter = DykstraPlotter(output_dir=str(output_dir))
    plotter.plot_convergence_comparison(
        results=results,
        labels=labels,
        max_iter=max_iter,
        filename=filename,
        show=False,
    )
    print(f"    [{solver_label}] saved selected-outer convergence plot: {output_dir / filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze inner Dykstra early termination behaviour from "
            "full-run JSON/NPZ artifacts."
        )
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Absolute/relative path to full-run JSON artifact.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to auto-select the latest matching full-run JSON.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(_default_results_dir()),
        help="Directory used when --seed is provided.",
    )
    parser.add_argument(
        "--sample-full-errors",
        type=int,
        default=5,
        help="How many full-budget (non-early) outer-iteration error traces to print.",
    )
    parser.add_argument(
        "--sample-early-errors",
        type=int,
        default=3,
        help="How many early-terminated (>0 iter) outer-iteration error traces to print.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="all",
        choices=("all", "vanilla", "fast"),
        help="Restrict analysis to one solver label or include all.",
    )
    parser.add_argument(
        "--component",
        type=int,
        default=None,
        help="Restrict analysis to one component dimension (e.g. 2).",
    )
    parser.add_argument(
        "--outer-iters",
        type=str,
        default=None,
        help="Comma-separated or bracketed outer iter list, e.g. '15,46,56' or '[15,46,56]'.",
    )
    parser.add_argument(
        "--prompt-outer-iters",
        action="store_true",
        help="After summary prints, prompt for an outer-iteration list to plot.",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=None,
        help="Directory for selected-outer convergence plots. Defaults next to full-run artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    json_path = _resolve_json_path(args.json, args.seed, results_dir)

    with open(json_path, "r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    npz_path = _resolve_npz_path(payload, json_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ path does not exist: {npz_path}")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid JSON: 'metadata' must be an object.")
    component_index = payload.get("npz_component_index", [])
    if not isinstance(component_index, list):
        raise ValueError("Invalid JSON: 'npz_component_index' must be a list.")

    selected_outer = _parse_outer_iter_list(args.outer_iters)
    if args.prompt_outer_iters:
        raw = input(
            "Enter outer iters to plot (comma-separated, empty to skip): "
        ).strip()
        selected_outer = _parse_outer_iter_list(raw)

    plot_output_dir = (
        Path(args.plot_output_dir).expanduser().resolve()
        if args.plot_output_dir is not None
        else json_path.parent / "analysis_plots"
    )

    print(f"JSON: {json_path}")
    print(f"NPZ:  {npz_path}")
    print(
        "Run metadata: "
        f"solver_mode={metadata.get('solver_mode')}, "
        f"seed={metadata.get('seed')}, "
        f"num_dimensions={metadata.get('num_dimensions')}, "
        f"max_outer_iter={metadata.get('max_outer_iter')}, "
        f"base_inner_iter={metadata.get('base_inner_iter')}, "
        f"max_inner_iters={metadata.get('max_inner_iters')}"
    )
    print()

    with np.load(npz_path, allow_pickle=True) as npz_data:
        for comp_meta in component_index:
            if not isinstance(comp_meta, dict):
                continue
            component_dim_raw = comp_meta.get("component_dim")
            if not isinstance(component_dim_raw, (int, np.integer)):
                continue
            component_dim = int(component_dim_raw)
            if args.component is not None and component_dim != args.component:
                continue

            solvers_meta = comp_meta.get("solvers", {})
            if not isinstance(solvers_meta, dict):
                continue

            print(f"Component {component_dim}")

            solver_labels = (
                [args.solver]
                if args.solver != "all"
                else sorted(str(k) for k in solvers_meta.keys())
            )
            for solver_label in solver_labels:
                solver_meta = solvers_meta.get(solver_label)
                if not isinstance(solver_meta, dict):
                    print(f"  [{solver_label}] unavailable in this artifact.")
                    continue

                summary = _print_component_solver_summary(
                    component_dim=component_dim,
                    solver_label=solver_label,
                    solver_meta=solver_meta,
                    npz_data=npz_data,
                )
                _print_error_samples(
                    solver_label=solver_label,
                    solver_meta=solver_meta,
                    summary=summary,
                    npz_data=npz_data,
                    sample_full_errors=max(int(args.sample_full_errors), 0),
                    sample_early_errors=max(int(args.sample_early_errors), 0),
                )
                _plot_selected_outer_iterations(
                    output_dir=plot_output_dir,
                    json_path=json_path,
                    component_dim=component_dim,
                    solver_label=solver_label,
                    selected_outer=selected_outer,
                    solver_meta=solver_meta,
                    npz_data=npz_data,
                )
            print()


if __name__ == "__main__":
    main()
