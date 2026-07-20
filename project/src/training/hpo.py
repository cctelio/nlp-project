"""Small sequential hyperparameter optimization runner."""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

from src.training.train_sft import materialize_sft_data_for_config, train_from_config
from src.utils.config import deep_update, dump_yaml, load_yaml
from src.utils.logging import read_json, write_json
from src.utils.paths import resolve_path
from src.utils.reproducibility import stable_int_seed


def _flatten_parameters(parameters: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    flattened: list[tuple[str, list[Any]]] = []
    for key, spec in parameters.items():
        if isinstance(spec, dict) and "values" in spec:
            values = spec["values"]
        elif isinstance(spec, list):
            values = spec
        else:
            values = [spec]
        flattened.append((key, list(values)))
    return flattened


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _trial_configs(base_config: dict[str, Any], sweep_config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = _flatten_parameters(sweep_config.get("parameters", {}))
    method = sweep_config.get("method", "grid")
    seed = int(sweep_config.get("seed", base_config.get("seed", 42)))

    all_assignments = []
    for values in itertools.product(*[values for _, values in parameters]):
        all_assignments.append(dict(zip([key for key, _ in parameters], values, strict=True)))

    if method == "random":
        rng = random.Random(stable_int_seed(seed, "hpo-random"))
        rng.shuffle(all_assignments)
        all_assignments = all_assignments[: int(sweep_config.get("trials", len(all_assignments)))]
    elif method != "grid":
        raise ValueError(f"Unsupported HPO method: {method}")

    configs: list[dict[str, Any]] = []
    for trial_index, assignment in enumerate(all_assignments):
        trial_config = deep_update({}, base_config)
        for key, value in assignment.items():
            _set_dotted(trial_config, key, value)
        strategy = trial_config.get("tokenizer", {}).get("strategy", "default")
        trial_config.setdefault("training", {})
        trial_config["training"]["run_name"] = f"hpo-{strategy}-trial{trial_index:03d}"
        trial_config.setdefault("hpo", {})
        trial_config["hpo"].update({"trial_index": trial_index, "assignment": assignment})
        configs.append(trial_config)
    return configs


def run_hpo(base_config: dict[str, Any], sweep_config: dict[str, Any], output_path: str | Path | None = None) -> dict[str, Any]:
    """Run a sequential grid/random sweep and select the best validation metric."""
    metric_config = sweep_config.get("metric", {})
    metric_name = metric_config.get("name", "validation/tanimoto_similarity_mean")
    goal = metric_config.get("goal", "maximize")
    primary_metric = {"name": metric_name, "goal": goal}
    metric_specs = [primary_metric] + list(sweep_config.get("tie_breakers", []))
    trial_results: list[dict[str, Any]] = []

    for trial_config in _trial_configs(base_config, sweep_config):
        if trial_config.get("data", {}).get("precompute_sft_per_trial", False):
            strategy = trial_config.get("tokenizer", {}).get("strategy", "default")
            trial_name = trial_config.get("training", {}).get("run_name", f"hpo-{strategy}")
            sft_output_dir = Path(trial_config.get("data", {}).get("sft_hpo_dir", "results/data/sft_hpo")) / trial_name
            sft_paths = materialize_sft_data_for_config(trial_config, sft_output_dir)
            trial_config.setdefault("data", {})
            trial_config["data"].update(
                {
                    "sft_train_path": sft_paths["train_path"],
                    "sft_validation_path": sft_paths["validation_path"],
                    "sft_test_path": sft_paths["test_path"],
                    "representation_path": sft_paths["representation_path"],
                }
            )
        run_dir = train_from_config(trial_config)
        metrics_path = run_dir / "metrics.json"
        metrics = read_json(metrics_path) if metrics_path.exists() else read_json(run_dir / "train_metrics.json")
        trial_results.append(
            {
                "run_dir": str(run_dir),
                "trial_index": trial_config.get("hpo", {}).get("trial_index"),
                "assignment": trial_config.get("hpo", {}).get("assignment", {}),
                "metric_name": metric_name,
                "metric_value": metrics.get(metric_name),
                "metrics": metrics,
            }
        )

    scored = [result for result in trial_results if result.get("metric_value") is not None]

    def score_tuple(result: dict[str, Any]) -> tuple[float, ...]:
        scores = []
        metrics = result.get("metrics", {})
        for spec in metric_specs:
            name = spec.get("name")
            spec_goal = spec.get("goal", "maximize")
            value = metrics.get(name)
            if value is None:
                value = float("-inf") if spec_goal == "maximize" else float("inf")
            scores.append(float(value) if spec_goal == "maximize" else -float(value))
        return tuple(scores)

    best = sorted(scored, key=score_tuple, reverse=True)[0] if scored else None
    summary = {
        "metric": {"name": metric_name, "goal": goal},
        "tie_breakers": sweep_config.get("tie_breakers", []),
        "best": best,
        "trials": trial_results,
    }
    if output_path:
        write_json(resolve_path(output_path), summary)
    return summary


def run_hpo_from_files(config_path: str | Path, tokenizer_config_path: str | Path, sweep_config_path: str | Path) -> dict[str, Any]:
    """Load config files and run HPO."""
    from src.utils.config import load_experiment_config

    base_config = load_experiment_config(config_path, tokenizer_config_path)
    sweep_config = load_yaml(sweep_config_path)
    output_path = sweep_config.get(
        "output_path",
        f"results/hpo/{base_config.get('tokenizer', {}).get('strategy', 'default')}_hpo_results.json",
    )
    Path(resolve_path(output_path)).parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(resolve_path(output_path).with_suffix(".yaml"), {"base_config": base_config, "sweep_config": sweep_config})
    return run_hpo(base_config, sweep_config, output_path=output_path)
