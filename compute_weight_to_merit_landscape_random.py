"""Train the architecture ablation and plot matched FSNet merit landscapes.

The three commands are:

* ``train``: train missing architecture checkpoints.
* ``visualize``: compute/plot landscapes from existing checkpoints.
* ``all``: train missing checkpoints and then visualize all landscapes.

All architectures use four hidden layers of width 64. Landscape direction
seeds, evaluation examples, perturbation radius, and color normalization are
shared across architectures so that corresponding plots are comparable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

# Keep Matplotlib's cache inside a writable temporary location on managed or
# headless systems where the user's normal config directory may be read-only.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "fsnet-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from utils.trainer import Evaluator, Trainer, create_model, load_instance


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ARCHITECTURES = ("MLP", "ICNN", "ResMLP")
HIDDEN_DIM = 64
NUM_LAYERS = 4
DEFAULT_DIRECTION_SEEDS = (0, 1, 2, 3, 4)
Y_DIRECTION_SEED_OFFSET = 1_000_003
LANDSCAPE_FORMAT_VERSION = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(base_config: Mapping, args: argparse.Namespace, architecture: str) -> dict:
    config = copy.deepcopy(dict(base_config))
    config.update(
        {
            "seed": args.training_seed,
            "method": "FSNet",
            "prob_type": args.prob_type,
            "prob_name": args.prob_name,
            "prob_size": list(args.prob_size),
            "network": architecture,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "ablation": False,
            "progress_bar": not args.no_progress_bar,
        }
    )

    scalar_overrides = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "lr": args.lr,
        "dropout": args.dropout,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            config[key] = value

    if args.fsnet_scale is not None:
        config["FSNet"]["scale"] = args.fsnet_scale
    if args.max_diff_iter is not None:
        config["FSNet"]["max_diff_iter"] = args.max_diff_iter
    return config


def checkpoint_directory(args: argparse.Namespace, architecture: str) -> Path:
    problem_size = "-".join(str(value) for value in args.prob_size)
    return (
        Path(args.checkpoint_root)
        / args.prob_type
        / args.prob_name
        / problem_size
        / f"{architecture}_{NUM_LAYERS}x{HIDDEN_DIM}"
        / f"train_seed{args.training_seed}"
    )


def checkpoint_path(args: argparse.Namespace, architecture: str) -> Path:
    return checkpoint_directory(args, architecture) / f"model_seed{args.training_seed}.pt"


def validate_checkpoint(
    checkpoint: Mapping,
    args: argparse.Namespace,
    architecture: str,
    expected_config: Mapping,
) -> None:
    config = checkpoint.get("config", {})
    expected = {
        "method": "FSNet",
        "prob_type": args.prob_type,
        "prob_name": args.prob_name,
        "network": architecture,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "seed": args.training_seed,
        "prob_size": list(args.prob_size),
        "num_epochs": expected_config["num_epochs"],
        "batch_size": expected_config["batch_size"],
        "val_size": expected_config["val_size"],
        "test_size": expected_config["test_size"],
        "lr": expected_config["lr"],
        "dropout": expected_config["dropout"],
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    expected_fsnet = expected_config["FSNet"]
    actual_fsnet = config.get("FSNet", {})
    # val_tol is intentionally omitted: the current trainer decays and mutates
    # it during training before writing the checkpoint.
    for key in (
        "obj_weight",
        "dist_weight",
        "eq_pen_weight",
        "ineq_pen_weight",
        "test_val_tol",
        "memory_size",
        "max_iter",
        "max_diff_iter",
        "scale",
    ):
        if actual_fsnet.get(key) != expected_fsnet.get(key):
            mismatches[f"FSNet.{key}"] = (
                actual_fsnet.get(key),
                expected_fsnet.get(key),
            )

    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"Checkpoint configuration mismatch for {architecture}: {details}")


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location=DEVICE, weights_only=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_signature(config: Mapping) -> str:
    relevant_config = {
        "seed": config["seed"],
        "prob_type": config["prob_type"],
        "prob_name": config["prob_name"],
        "prob_size": config["prob_size"],
        "test_size": config["test_size"],
        "FSNet": config["FSNet"],
    }
    serialized = json.dumps(relevant_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def train_architecture(
    base_config: Mapping,
    args: argparse.Namespace,
    architecture: str,
) -> Path:
    config = build_config(base_config, args, architecture)
    path = checkpoint_path(args, architecture)
    if path.exists() and not args.force_train:
        checkpoint = load_checkpoint(path)
        validate_checkpoint(checkpoint, args, architecture, config)
        print(f"Reusing {architecture} checkpoint: {path}", flush=True)
        return path

    seed_everything(args.training_seed)
    problem, _unused_default_save_dir = load_instance(config)
    save_dir = checkpoint_directory(args, architecture)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nTraining {architecture}: {NUM_LAYERS} hidden layers x {HIDDEN_DIM} units "
        f"on {args.prob_type}/{args.prob_name}",
        flush=True,
    )
    trainer = Trainer(data=problem, config=config, save_dir=str(save_dir))
    trainer.train()
    if not path.exists():
        raise RuntimeError(f"Training finished but did not create checkpoint: {path}")
    return path


def clone_weights(model: torch.nn.Module) -> List[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def get_random_weights_like_params(
    parameters: Sequence[torch.Tensor], seed: int
) -> List[torch.Tensor]:
    generator = torch.Generator(device=parameters[0].device)
    generator.manual_seed(seed)
    return [
        torch.randn(
            parameter.shape,
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        for parameter in parameters
    ]


def normalize_directions_for_weights(
    direction: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    norm: str = "filter",
    ignore: str = "biasbn",
) -> None:
    if len(direction) != len(weights):
        raise ValueError("Direction and weight lists must have the same length")

    for delta, weight in zip(direction, weights):
        if delta.ndim <= 1:
            if ignore == "biasbn":
                delta.zero_()
            elif norm == "weight":
                delta.mul_(weight)
            else:
                delta.mul_(weight.norm() / (delta.norm() + 1e-12))
            continue

        if norm == "filter":
            delta_flat = delta.reshape(delta.shape[0], -1)
            weight_flat = weight.reshape(weight.shape[0], -1)
            scale = weight_flat.norm(dim=1) / (delta_flat.norm(dim=1) + 1e-12)
            delta.mul_(scale.reshape((-1,) + (1,) * (delta.ndim - 1)))
        elif norm == "layer":
            delta.mul_(weight.norm() / (delta.norm() + 1e-12))
        elif norm == "weight":
            delta.mul_(weight)
        elif norm == "direction":
            delta.div_(delta.norm() + 1e-12)
        else:
            raise ValueError(f"Unknown direction normalization: {norm}")


def create_random_direction(
    model: torch.nn.Module,
    seed: int,
    norm: str = "filter",
) -> List[torch.Tensor]:
    parameters = list(model.parameters())
    direction = get_random_weights_like_params(parameters, seed=seed)
    normalize_directions_for_weights(direction, parameters, norm=norm)
    return direction


@torch.no_grad()
def set_weights(
    model: torch.nn.Module,
    base_weights: Sequence[torch.Tensor],
    directions: Sequence[Sequence[torch.Tensor]] | None = None,
    step: Tuple[float, float] | None = None,
) -> None:
    if directions is None:
        for parameter, base in zip(model.parameters(), base_weights):
            parameter.copy_(base)
        return

    if step is None or len(directions) != 2:
        raise ValueError("A two-dimensional step requires exactly two directions")
    x_step, y_step = float(step[0]), float(step[1])
    x_direction, y_direction = directions
    for parameter, base, delta_x, delta_y in zip(
        model.parameters(), base_weights, x_direction, y_direction
    ):
        parameter.copy_(base + x_step * delta_x + y_step * delta_y)


def merit_values(
    problem,
    input_batch: torch.Tensor,
    prediction: torch.Tensor,
    equality_weight: float,
    inequality_weight: float,
) -> torch.Tensor:
    """Return the per-example objective-plus-L1-violation merit values."""
    objective = problem.obj_fn(prediction)
    equality_violation = problem.eq_resid(
        input_batch, prediction
    ).abs().sum(dim=1)
    inequality_violation = problem.ineq_resid(
        input_batch, prediction
    ).abs().sum(dim=1)
    return (
        objective
        + equality_weight * equality_violation
        + inequality_weight * inequality_violation
    )


def make_merit_evaluator(
    problem,
    config: Mapping,
    equality_weight: float,
    inequality_weight: float,
) -> Callable[[torch.nn.Module, DataLoader], float]:
    evaluator = Evaluator(problem, "FSNet", config)

    @torch.no_grad()
    def evaluate_merit(model: torch.nn.Module, data_loader: DataLoader) -> float:
        model.eval()
        total_merit = 0.0
        total_samples = 0
        for input_batch, _target_batch in data_loader:
            input_batch = input_batch.to(DEVICE, non_blocking=True)
            prediction = model(input_batch)
            scaled_prediction = problem.scale(prediction)
            final_prediction = evaluator._post_process_predictions(
                input_batch, scaled_prediction
            )

            merit = merit_values(
                problem,
                input_batch,
                final_prediction,
                equality_weight,
                inequality_weight,
            )
            total_merit += merit.sum().item()
            total_samples += input_batch.shape[0]
        return total_merit / max(1, total_samples)

    return evaluate_merit


@torch.no_grad()
def compute_merit_surface_2d(
    model: torch.nn.Module,
    test_loader: DataLoader,
    eval_merit: Callable[[torch.nn.Module, DataLoader], float],
    *,
    radius: float,
    grid_size: int,
    direction_seed: int,
    norm: str,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(DEVICE).eval()
    base_weights = clone_weights(model)
    x_seed = direction_seed
    y_seed = direction_seed + Y_DIRECTION_SEED_OFFSET
    direction_x = create_random_direction(model, seed=x_seed, norm=norm)
    direction_y = create_random_direction(model, seed=y_seed, norm=norm)

    coordinates = np.linspace(-radius, radius, grid_size, dtype=np.float64)
    merit = np.empty((grid_size, grid_size), dtype=np.float64)
    total = grid_size * grid_size
    completed = 0

    try:
        for row, y_coordinate in enumerate(coordinates):
            for column, x_coordinate in enumerate(coordinates):
                set_weights(
                    model,
                    base_weights,
                    directions=(direction_x, direction_y),
                    step=(x_coordinate, y_coordinate),
                )
                merit[row, column] = eval_merit(model, test_loader)
                completed += 1
                if verbose and (
                    completed == total or completed % max(1, total // 20) == 0
                ):
                    print(
                        f"  [{completed:>5}/{total}] direction_seed={direction_seed} "
                        f"x={x_coordinate:+.3f} y={y_coordinate:+.3f} "
                        f"merit={merit[row, column]:.6g}",
                        flush=True,
                    )
    finally:
        set_weights(model, base_weights)

    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    return x_grid, y_grid, merit


def surface_path(output_dir: Path, architecture: str, direction_seed: int) -> Path:
    return output_dir / f"merit_landscape_{architecture}_direction_seed{direction_seed}.npz"


def save_surface(
    path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    merit: np.ndarray,
    args: argparse.Namespace,
    architecture: str,
    direction_seed: int,
    checkpoint_digest: str,
    eval_signature: str,
) -> None:
    np.savez_compressed(
        path,
        X=x_grid,
        Y=y_grid,
        Z=merit,
        architecture=architecture,
        training_seed=args.training_seed,
        direction_seed=direction_seed,
        x_direction_seed=direction_seed,
        y_direction_seed=direction_seed + Y_DIRECTION_SEED_OFFSET,
        penalty_weight=args.penalty_weight,
        radius=args.radius,
        grid_size=args.grid_size,
        direction_norm=args.direction_norm,
        landscape_test_size=args.landscape_test_size,
        checkpoint_sha256=checkpoint_digest,
        evaluation_signature=eval_signature,
        landscape_format_version=LANDSCAPE_FORMAT_VERSION,
    )


def cached_surface_matches(
    surface: Mapping,
    args: argparse.Namespace,
    architecture: str,
    direction_seed: int,
    checkpoint_digest: str,
    eval_signature: str,
) -> bool:
    expected = {
        "architecture": architecture,
        "training_seed": args.training_seed,
        "direction_seed": direction_seed,
        "penalty_weight": args.penalty_weight,
        "radius": args.radius,
        "grid_size": args.grid_size,
        "direction_norm": args.direction_norm,
        "landscape_test_size": args.landscape_test_size,
        "checkpoint_sha256": checkpoint_digest,
        "evaluation_signature": eval_signature,
        "landscape_format_version": LANDSCAPE_FORMAT_VERSION,
    }
    for key, expected_value in expected.items():
        if key not in surface:
            return False
        actual = surface[key].item()
        if isinstance(expected_value, float):
            if not np.isclose(actual, expected_value):
                return False
        elif actual != expected_value:
            return False
    return True


def load_model_for_landscape(
    problem,
    args: argparse.Namespace,
    architecture: str,
    expected_config: Mapping,
) -> torch.nn.Module:
    path = checkpoint_path(args, architecture)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {architecture} checkpoint: {path}. Run the 'train' or 'all' command first."
        )
    checkpoint = load_checkpoint(path)
    validate_checkpoint(checkpoint, args, architecture, expected_config)
    model = create_model(problem, "FSNet", checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(DEVICE).eval()


def draw_surface(
    axis,
    surface: Mapping[str, np.ndarray],
    color_minimum: float,
    color_maximum: float,
    title: str,
    elevation: float,
    azimuth: float,
):
    raw_merit = surface["Z"]
    plot_values = np.ma.masked_invalid(raw_merit)
    surface_plot = axis.plot_surface(
        surface["X"],
        surface["Y"],
        plot_values,
        cmap="viridis",
        vmin=color_minimum,
        vmax=color_maximum,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    center = raw_merit[raw_merit.shape[0] // 2, raw_merit.shape[1] // 2]
    if np.isfinite(center):
        axis.scatter(
            [0.0],
            [0.0],
            [center],
            color="red",
            marker="x",
            s=35,
            linewidths=1.8,
            depthshade=False,
        )
    axis.set_title(title)
    axis.set_xlabel("direction 1")
    axis.set_ylabel("direction 2")
    axis.set_zlabel("raw merit")
    axis.set_xlim(float(surface["X"].min()), float(surface["X"].max()))
    axis.set_ylim(float(surface["Y"].min()), float(surface["Y"].max()))
    axis.set_zlim(color_minimum, color_maximum)
    axis.set_box_aspect((1.0, 1.0, 0.78))
    axis.zaxis.labelpad = 9
    axis.tick_params(axis="both", which="major", labelsize=8, pad=1)
    axis.tick_params(axis="z", which="major", labelsize=8, pad=2)
    axis.view_init(elev=elevation, azim=azimuth)
    return surface_plot


def save_figure(figure, path: Path, dpi: int) -> None:
    """Save a high-resolution figure without clipping 3D labels or colorbars."""
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.3,
        facecolor="white",
    )


def plot_surfaces(
    surfaces: Mapping[Tuple[str, int], Mapping[str, np.ndarray]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    all_values = np.concatenate([surface["Z"].ravel() for surface in surfaces.values()])
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size == 0:
        raise ValueError("All computed merit values are non-finite")
    global_minimum = float(finite_values.min())
    color_minimum = global_minimum
    color_maximum = float(finite_values.max())
    if np.isclose(color_minimum, color_maximum):
        color_maximum = color_minimum + 1.0
    colorbar_label = "raw merit"

    for (architecture, direction_seed), surface in surfaces.items():
        figure = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        surface_plot = draw_surface(
            axis,
            surface,
            color_minimum,
            color_maximum,
            f"{architecture}, direction seed {direction_seed}",
            args.elevation,
            args.azimuth,
        )
        figure.colorbar(surface_plot, ax=axis, label=colorbar_label, shrink=0.72, pad=0.1)
        save_figure(
            figure,
            output_dir / f"merit_landscape_{architecture}_direction_seed{direction_seed}.png",
            args.dpi,
        )
        plt.close(figure)

    for direction_seed in args.direction_seeds:
        figure, axes = plt.subplots(
            1,
            len(args.architectures),
            figsize=(5.8 * len(args.architectures), 5.4),
            constrained_layout=True,
            squeeze=False,
            subplot_kw={"projection": "3d"},
        )
        surface_plot = None
        for column, architecture in enumerate(args.architectures):
            surface_plot = draw_surface(
                axes[0, column],
                surfaces[(architecture, direction_seed)],
                color_minimum,
                color_maximum,
                architecture,
                args.elevation,
                args.azimuth,
            )
        figure.suptitle(f"Matched FSNet merit landscapes: direction seed {direction_seed}")
        figure.colorbar(
            surface_plot,
            ax=axes.ravel().tolist(),
            label=colorbar_label,
            shrink=0.7,
            pad=0.04,
        )
        save_figure(
            figure,
            output_dir / f"merit_landscape_comparison_direction_seed{direction_seed}.png",
            args.dpi,
        )
        plt.close(figure)

    figure, axes = plt.subplots(
        len(args.direction_seeds),
        len(args.architectures),
        figsize=(5.6 * len(args.architectures), 4.8 * len(args.direction_seeds)),
        constrained_layout=True,
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    surface_plot = None
    for row, direction_seed in enumerate(args.direction_seeds):
        for column, architecture in enumerate(args.architectures):
            surface_plot = draw_surface(
                axes[row, column],
                surfaces[(architecture, direction_seed)],
                color_minimum,
                color_maximum,
                f"{architecture}, direction seed {direction_seed}",
                args.elevation,
                args.azimuth,
            )
    figure.suptitle(
        f"FSNet architecture merit landscapes "
        f"(penalty weight={args.penalty_weight:g}, train seed={args.training_seed})"
    )
    figure.colorbar(
        surface_plot,
        ax=axes.ravel().tolist(),
        label=colorbar_label,
        shrink=0.65,
        pad=0.025,
    )
    save_figure(
        figure,
        output_dir / "merit_landscape_all_comparisons.png",
        args.dpi,
    )
    plt.close(figure)

    return {
        "global_merit_minimum": global_minimum,
        "global_merit_maximum": float(finite_values.max()),
        "color_minimum": color_minimum,
        "color_maximum": color_maximum,
        "color_transform": "none (raw merit)",
        "plot_projection": "3d",
        "view_elevation": args.elevation,
        "view_azimuth": args.azimuth,
    }


def surface_statistics(merit: np.ndarray, center_index: int) -> dict:
    finite_values = merit[np.isfinite(merit)]
    center = merit[center_index, center_index]
    return {
        "minimum": float(finite_values.min()) if finite_values.size else None,
        "maximum": float(finite_values.max()) if finite_values.size else None,
        "center": float(center) if np.isfinite(center) else None,
    }


def visualize_landscapes(base_config: Mapping, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One shared split, subset, loader, evaluator, and post-processing config is
    # deliberately reused for every architecture.
    evaluation_config = build_config(base_config, args, args.architectures[0])
    evaluation_config["progress_bar"] = False
    seed_everything(args.training_seed)
    problem, _unused_default_save_dir = load_instance(evaluation_config)

    test_dataset = problem.test_dataset
    requested_size = args.landscape_test_size
    if requested_size > 0 and requested_size < len(test_dataset):
        test_dataset = Subset(test_dataset, range(requested_size))
    actual_test_size = len(test_dataset)
    args.landscape_test_size = actual_test_size
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(args.landscape_batch_size, actual_test_size),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    evaluate_merit = make_merit_evaluator(
        problem,
        evaluation_config,
        equality_weight=args.penalty_weight,
        inequality_weight=args.penalty_weight,
    )
    eval_signature = evaluation_signature(evaluation_config)

    surfaces: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for architecture in args.architectures:
        architecture_config = build_config(base_config, args, architecture)
        model = load_model_for_landscape(
            problem, args, architecture, architecture_config
        )
        checkpoint_digest = file_sha256(checkpoint_path(args, architecture))
        for direction_seed in args.direction_seeds:
            path = surface_path(output_dir, architecture, direction_seed)
            if path.exists() and not args.force_landscape:
                with np.load(path, allow_pickle=False) as cached:
                    if cached_surface_matches(
                        cached,
                        args,
                        architecture,
                        direction_seed,
                        checkpoint_digest,
                        eval_signature,
                    ):
                        print(f"Reusing landscape data: {path}", flush=True)
                        surfaces[(architecture, direction_seed)] = {
                            "X": cached["X"].copy(),
                            "Y": cached["Y"].copy(),
                            "Z": cached["Z"].copy(),
                        }
                        continue

            print(
                f"\nComputing {architecture} landscape for direction seed {direction_seed} "
                f"on {actual_test_size} examples",
                flush=True,
            )
            x_grid, y_grid, merit = compute_merit_surface_2d(
                model,
                test_loader,
                evaluate_merit,
                radius=args.radius,
                grid_size=args.grid_size,
                direction_seed=direction_seed,
                norm=args.direction_norm,
                verbose=not args.quiet_surface,
            )
            save_surface(
                path,
                x_grid,
                y_grid,
                merit,
                args,
                architecture,
                direction_seed,
                checkpoint_digest,
                eval_signature,
            )
            surfaces[(architecture, direction_seed)] = {
                "X": x_grid,
                "Y": y_grid,
                "Z": merit,
            }
            print(f"Saved landscape data: {path}", flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_metadata = plot_surfaces(surfaces, args, output_dir)
    manifest = {
        "problem_type": args.prob_type,
        "problem_name": args.prob_name,
        "problem_size": list(args.prob_size),
        "architectures": list(args.architectures),
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "training_seed": args.training_seed,
        "direction_seeds": list(args.direction_seeds),
        "direction_seed_pairs": {
            str(seed): [seed, seed + Y_DIRECTION_SEED_OFFSET]
            for seed in args.direction_seeds
        },
        "direction_norm": args.direction_norm,
        "radius": args.radius,
        "grid_size": args.grid_size,
        "landscape_test_size": actual_test_size,
        "landscape_batch_size": args.landscape_batch_size,
        "equality_penalty_weight": args.penalty_weight,
        "inequality_penalty_weight": args.penalty_weight,
        "device": str(DEVICE),
        "evaluation_signature": eval_signature,
        "landscape_format_version": LANDSCAPE_FORMAT_VERSION,
        **plot_metadata,
        "surface_statistics": {
            f"{architecture}/direction_seed{seed}": surface_statistics(
                surface["Z"], args.grid_size // 2
            )
            for (architecture, seed), surface in surfaces.items()
        },
    }
    manifest_path = output_dir / "merit_landscape_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
    print(f"\nSaved comparable plots and manifest under: {output_dir}", flush=True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and visualize matched FSNet architecture merit landscapes."
    )
    parser.add_argument("command", choices=("train", "visualize", "all"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )
    parser.add_argument("--prob-type", default="nonsmooth_nonconvex")
    parser.add_argument("--prob-name", default="socp")
    parser.add_argument(
        "--prob-size",
        nargs=4,
        type=int,
        default=[100, 50, 50, 10000],
        metavar=("VARIABLES", "INEQUALITIES", "EQUALITIES", "EXAMPLES"),
    )
    parser.add_argument("--training-seed", type=int, default=2025)
    parser.add_argument(
        "--direction-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_DIRECTION_SEEDS),
    )

    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--fsnet-scale", type=float)
    parser.add_argument("--max-diff-iter", type=int)
    parser.add_argument("--no-progress-bar", action="store_true")

    parser.add_argument("--penalty-weight", type=float, default=1e4)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=31,
        help="Odd number of evaluated points per direction (default: 31)",
    )
    parser.add_argument(
        "--direction-norm",
        choices=("filter", "layer", "weight", "direction"),
        default="filter",
    )
    parser.add_argument("--landscape-test-size", type=int, default=128)
    parser.add_argument("--landscape-batch-size", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--azimuth", type=float, default=-60.0)
    parser.add_argument("--quiet-surface", action="store_true")

    parser.add_argument("--checkpoint-root", default="results/landscape_ablation")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-landscape", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.architectures = list(dict.fromkeys(args.architectures))
    args.direction_seeds = list(dict.fromkeys(args.direction_seeds))
    if args.grid_size < 3 or args.grid_size % 2 == 0:
        parser.error("--grid-size must be an odd integer of at least 3")
    if args.radius <= 0:
        parser.error("--radius must be positive")
    if args.penalty_weight < 0:
        parser.error("--penalty-weight must be nonnegative")
    if args.landscape_test_size == 0 or args.landscape_test_size < -1:
        parser.error("--landscape-test-size must be -1 (all) or a positive integer")
    if args.landscape_batch_size <= 0:
        parser.error("--landscape-batch-size must be positive")
    for name in ("num_epochs", "batch_size", "val_size", "test_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    with open(args.config, encoding="utf-8") as file:
        base_config = yaml.safe_load(file)

    print(
        f"FSNet landscape ablation: architectures={args.architectures}, "
        f"shape={NUM_LAYERS}x{HIDDEN_DIM}, training_seed={args.training_seed}, "
        f"direction_seeds={args.direction_seeds}",
        flush=True,
    )
    if args.command in {"train", "all"}:
        for architecture in args.architectures:
            train_architecture(base_config, args, architecture)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if args.command in {"visualize", "all"}:
        visualize_landscapes(base_config, args)


if __name__ == "__main__":
    main()
