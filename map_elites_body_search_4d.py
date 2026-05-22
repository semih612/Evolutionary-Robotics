"""
Stage 1 — Sparse 4D MAP-Elites Body Search with Fixed Baseline Controller
==========================================================================

Purpose
-------
This script keeps the controller parameters fixed from a baseline NSGA-II run
and uses MAP-Elites to search only the robot body parameters.

Baseline genotype:
    x_baseline = [controller_params | body_params]

MAP-Elites individual:
    body_params only, length = 8

Evaluation genotype:
    x_candidate = [fixed_baseline_controller | candidate_body_params]

4D descriptor:
    descriptor = total physical length of each leg: [front-left, front-right, back-left, back-right]

Because even a 4D archive can grow quickly, this implementation uses a sparse
Python dictionary:
    archive[cell_tuple] = elite

Each cell tuple has 4 integer bin indices:
    cell = (front_left_bin, front_right_bin, back_left_bin, back_right_bin)

Default fitness:
    fitness = flat_score + ice_score + hill_score

This matches the common baseline scalar selection rule used in many NSGA-II
training scripts where the best individual is chosen by fitness.sum(axis=1).

Outputs
-------
output_dir/
    archive_cells.npy
    archive_body.npy
    archive_scores.npy
    archive_fitness.npy
    archive_descriptor.npy
    archive_summary.txt
    archive_projection_avg_std.png
    archive_pairwise_projections_4d.png
    selected_bodies/
        body_00.npy ... body_09.npy
        full_genotype_fixed_controller_00.npy ...
        x_best.npy

Usage examples
--------------
Smoke test:
    python map_elites_body_search_4d.py \
        --main_module final_project_train_no_map_elites \
        --baseline_dir results/nsga_baseline_no_map_elites \
        --output_dir results/map_elites_body_stage1_4d \
        --n_bins_per_dim 8 \
        --n_init 32 \
        --n_iterations 200 \
        --n_repeats 1 \
        --n_steps 200 \
        --fitness_mode sum \
        --evaluate_after

Larger run:
    python map_elites_body_search_4d.py \
        --main_module final_project_train_no_map_elites \
        --baseline_dir results/nsga_baseline_no_map_elites \
        --output_dir results/map_elites_body_stage1_4d \
        --n_bins_per_dim 8 \
        --n_init 256 \
        --n_iterations 3000 \
        --n_repeats 2 \
        --n_steps 300 \
        --fitness_mode sum \
        --evaluate_after
"""

from __future__ import annotations

import argparse
import importlib
import os
from os.path import join
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

# Useful for headless/remote Linux MuJoCo rendering.
os.environ.setdefault("MUJOCO_GL", "egl")


# ---------------------------------------------------------------------------
# Imports from main training module
# ---------------------------------------------------------------------------

def import_main_module(module_name: str):
    """Import the user's main training module."""
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Body genotype and 4D descriptor helpers
# ---------------------------------------------------------------------------

def body_genotype_to_lengths(body_params: np.ndarray) -> np.ndarray:
    """
    Same body mapping used in FinalWorld.geno2pheno:
        genotype value g in [-1, 1] -> length in [0.1, 0.6]
    """
    body_params = np.asarray(body_params, dtype=float)
    return (body_params + 1.0) / 4.0 + 0.1


def compute_body_descriptor(body_params: np.ndarray) -> np.ndarray:
    """
    4D morphology descriptor: one total length per leg.

    The body genotype still has 8 parameters:
        [front_left_upper, front_left_lower,
         front_right_upper, front_right_lower,
         back_left_upper, back_left_lower,
         back_right_upper, back_right_lower]

    The descriptor compresses this into 4 leg-level quantities:
        [front_left_total, front_right_total, back_left_total, back_right_total]

    This is a compromise between the 2D descriptor and the 8D descriptor:
    it keeps leg-specific information but avoids the extreme sparsity of 8D.
    """
    lengths = body_genotype_to_lengths(body_params)

    fl_upper, fl_lower, fr_upper, fr_lower, bl_upper, bl_lower, br_upper, br_lower = lengths

    front_left_total = fl_upper + fl_lower
    front_right_total = fr_upper + fr_lower
    back_left_total = bl_upper + bl_lower
    back_right_total = br_upper + br_lower

    return np.array([
        float(front_left_total),
        float(front_right_total),
        float(back_left_total),
        float(back_right_total),
    ])


def descriptor_to_cell(
    descriptor: np.ndarray,
    n_bins_per_dim: int = 8,
    total_leg_range: tuple[float, float] = (0.2, 1.2),
) -> tuple[int, ...]:
    """
    Convert 4 total leg lengths into a 4D MAP-Elites cell.

    Each total leg length is in approximately [0.2, 1.2], because each
    leg is made of two segments and each segment is in [0.1, 0.6].

    Example:
        descriptor = [0.65, 0.70, 0.55, 0.80]
        cell       = (3, 4, 2, 5)
    """
    descriptor = np.asarray(descriptor, dtype=float)

    bins = (
        (descriptor - total_leg_range[0])
        / (total_leg_range[1] - total_leg_range[0])
        * n_bins_per_dim
    ).astype(int)

    bins = np.clip(bins, 0, n_bins_per_dim - 1)
    return tuple(int(b) for b in bins)


def scalar_fitness(scores: np.ndarray, mode: str = "sum") -> float:
    """
    Convert [flat, ice, hill] scores into one scalar archive fitness.

    mode='sum'            : flat + ice + hill, matches baseline sum selection
    mode='min'            : worst-terrain score, conservative generalist metric
    mode='mean_minus_std' : mean - 0.5*std, smoother balanced metric
    mode='mixed'          : 0.7*min + 0.3*mean
    """
    scores = np.asarray(scores, dtype=float)

    if mode == "sum":
        return float(np.sum(scores))
    if mode == "min":
        return float(np.min(scores))
    if mode == "mean_minus_std":
        return float(np.mean(scores) - 0.5 * np.std(scores))
    if mode == "mixed":
        return float(0.7 * np.min(scores) + 0.3 * np.mean(scores))

    raise ValueError(f"Unknown fitness mode: {mode}")


def mutate_body(
    body_params: np.ndarray,
    sigma: float,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Gaussian mutation on body parameters only."""
    body_params = np.asarray(body_params, dtype=float)
    child = body_params + sigma * np.random.randn(*body_params.shape)
    return np.clip(child, bounds[0], bounds[1])


# ---------------------------------------------------------------------------
# Baseline loading and evaluation
# ---------------------------------------------------------------------------

def get_last_checkpoint_dir_fallback(path: str) -> str | None:
    """
    Fallback if the imported main module does not expose get_last_checkpoint_dir.
    Looks for integer-named checkpoint subdirectories and returns the largest.
    """
    if not os.path.isdir(path):
        return None

    gens = []
    for name in os.listdir(path):
        full = join(path, name)
        if os.path.isdir(full) and name.isdigit():
            gens.append((int(name), full))

    if not gens:
        return None

    return sorted(gens, key=lambda x: x[0])[-1][1]


def load_baseline_genotype(baseline_dir: str, main_module: Any) -> np.ndarray:
    """
    Load x_best.npy from either the last checkpoint folder or baseline_dir.
    """
    last_gen = None

    if hasattr(main_module, "get_last_checkpoint_dir"):
        last_gen = main_module.get_last_checkpoint_dir(baseline_dir)

    if last_gen is None:
        last_gen = get_last_checkpoint_dir_fallback(baseline_dir)

    candidate_dirs = []
    if last_gen is not None:
        candidate_dirs.append(last_gen)
    candidate_dirs.append(baseline_dir)

    for d in candidate_dirs:
        p = join(d, "x_best.npy")
        if os.path.isfile(p):
            print(f"Loaded baseline x_best.npy from: {p}")
            return np.load(p, allow_pickle=True)

    raise FileNotFoundError(
        f"Could not find x_best.npy in '{baseline_dir}' or its last checkpoint."
    )


def evaluate_body_with_fixed_controller(
    world: Any,
    body_params: np.ndarray,
    fixed_controller_params: np.ndarray,
    n_repeats: int,
    n_steps: int,
) -> np.ndarray:
    """
    Evaluate candidate body using the fixed baseline controller.

    The full genotype is reconstructed only for evaluation:
        [fixed_controller_params | body_params]
    """
    full_genotype = np.concatenate([fixed_controller_params, body_params])
    return world.evaluate_individual(
        full_genotype,
        n_repeats=n_repeats,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Sparse archive utilities
# ---------------------------------------------------------------------------

def try_insert_sparse(
    archive: dict[tuple[int, ...], dict[str, Any]],
    body: np.ndarray,
    scores: np.ndarray,
    n_bins_per_dim: int,
    fitness_mode: str,
) -> tuple[bool, tuple[int, ...], float]:
    """Insert candidate into archive if its cell is empty or it is better."""
    descriptor = compute_body_descriptor(body)
    cell = descriptor_to_cell(descriptor, n_bins_per_dim=n_bins_per_dim)
    fitness = scalar_fitness(scores, mode=fitness_mode)

    if cell not in archive or fitness > archive[cell]["fitness"]:
        archive[cell] = {
            "body": body.copy(),
            "scores": scores.copy(),
            "fitness": float(fitness),
            "descriptor": descriptor.copy(),
        }
        return True, cell, fitness

    return False, cell, fitness


def save_sparse_archive(output_dir: str, archive: dict[tuple[int, ...], dict[str, Any]]) -> None:
    """Save sparse archive into compact arrays."""
    os.makedirs(output_dir, exist_ok=True)

    if len(archive) == 0:
        np.save(join(output_dir, "archive_cells.npy"), np.empty((0, 4), dtype=int))
        np.save(join(output_dir, "archive_body.npy"), np.empty((0, 8)))
        np.save(join(output_dir, "archive_scores.npy"), np.empty((0, 3)))
        np.save(join(output_dir, "archive_fitness.npy"), np.empty((0,)))
        np.save(join(output_dir, "archive_descriptor.npy"), np.empty((0, 4)))
        return

    cells = list(archive.keys())
    cells_array = np.asarray(cells, dtype=int)
    bodies = np.asarray([archive[c]["body"] for c in cells], dtype=float)
    scores = np.asarray([archive[c]["scores"] for c in cells], dtype=float)
    fitness = np.asarray([archive[c]["fitness"] for c in cells], dtype=float)
    descriptors = np.asarray([archive[c]["descriptor"] for c in cells], dtype=float)

    np.save(join(output_dir, "archive_cells.npy"), cells_array)
    np.save(join(output_dir, "archive_body.npy"), bodies)
    np.save(join(output_dir, "archive_scores.npy"), scores)
    np.save(join(output_dir, "archive_fitness.npy"), fitness)
    np.save(join(output_dir, "archive_descriptor.npy"), descriptors)


def write_sparse_archive_summary(
    output_dir: str,
    archive: dict[tuple[int, ...], dict[str, Any]],
    n_bins_per_dim: int,
    fitness_mode: str,
) -> None:
    """Write a human-readable archive summary."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = join(output_dir, "archive_summary.txt")

    total_possible_cells = n_bins_per_dim ** 4

    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Sparse 4D MAP-Elites Body Search Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Fitness mode        : {fitness_mode}\n")
        f.write(f"Bins per dimension  : {n_bins_per_dim}\n")
        f.write(f"Possible cells      : {total_possible_cells}\n")
        f.write(f"Occupied cells      : {len(archive)}\n")
        f.write(f"Archive coverage    : {len(archive) / total_possible_cells:.8f}\n")

        if len(archive) > 0:
            best_cell, best_elite = max(
                archive.items(),
                key=lambda item: item[1]["fitness"],
            )
            f.write("\nBest archive elite:\n")
            f.write(f"  Cell       : {best_cell}\n")
            f.write(f"  Fitness    : {best_elite['fitness']:.3f}\n")
            f.write(
                f"  Scores     : flat={best_elite['scores'][0]:.3f}, "
                f"ice={best_elite['scores'][1]:.3f}, "
                f"hill={best_elite['scores'][2]:.3f}\n"
            )
            f.write(f"  Body geno  : {best_elite['body']}\n")
            f.write(f"  Body len   : {best_elite['descriptor']}\n")

    print(f"Saved archive summary to: {summary_path}")


def plot_sparse_archive_projection(
    output_dir: str,
    archive: dict[tuple[int, ...], dict[str, Any]],
    n_bins_per_dim: int,
) -> None:
    """
    Optional 2D projection for visualization.

    Since the true archive is 4D, a heatmap of all dimensions is not directly shown here.
    This plot projects elites into:
        x = average total leg length
        y = standard deviation of total leg lengths
    and colors them by archive fitness.
    """
    if len(archive) == 0:
        return

    descriptors = np.asarray([elite["descriptor"] for elite in archive.values()])
    fitness = np.asarray([elite["fitness"] for elite in archive.values()])

    avg_len = descriptors.mean(axis=1)
    std_len = descriptors.std(axis=1)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(avg_len, std_len, c=fitness, s=28, alpha=0.85)
    plt.colorbar(sc, label="Archive fitness")
    plt.xlabel("Average total leg length")
    plt.ylabel("Total leg length standard deviation")
    plt.title("Sparse 4D MAP-Elites Archive Projection")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    save_path = join(output_dir, "archive_projection_avg_std.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved 2D archive projection to: {save_path}")



def plot_archive_pairwise_projections_4d(
    output_dir: str,
    archive: dict[tuple[int, ...], dict[str, Any]],
    n_bins_per_dim: int,
    descriptor_names: list[str] | None = None,
) -> None:
    """
    Plot 2D pairwise projections of the sparse 4D MAP-Elites archive.

    The true archive cell is 4D:
        (front_left_bin, front_right_bin, back_left_bin, back_right_bin)

    Each subplot chooses two descriptor dimensions for the axes and projects
    away the other two dimensions by keeping the best fitness found for each
    pair of bins. This gives an interpretable view of which leg-length
    combinations tend to contain high-performing elites.
    """
    if len(archive) == 0:
        print("Archive is empty; skipping 4D pairwise projection plot.")
        return

    if descriptor_names is None:
        descriptor_names = [
            "Front-left total",
            "Front-right total",
            "Back-left total",
            "Back-right total",
        ]

    pairs = [
        (0, 1),  # front-left vs front-right
        (0, 2),  # front-left vs back-left
        (0, 3),  # front-left vs back-right
        (1, 2),  # front-right vs back-left
        (1, 3),  # front-right vs back-right
        (2, 3),  # back-left vs back-right
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax, (d1, d2) in zip(axes, pairs):
        projection = np.full((n_bins_per_dim, n_bins_per_dim), np.nan)

        for cell, elite in archive.items():
            i = int(cell[d1])
            j = int(cell[d2])
            fit = float(elite["fitness"])

            if np.isnan(projection[i, j]) or fit > projection[i, j]:
                projection[i, j] = fit

        im = ax.imshow(
            projection.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_xlabel(f"{descriptor_names[d1]} bin")
        ax.set_ylabel(f"{descriptor_names[d2]} bin")
        ax.set_title(f"{descriptor_names[d1]} vs {descriptor_names[d2]}")
        ax.set_xticks(range(n_bins_per_dim))
        ax.set_yticks(range(n_bins_per_dim))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Best archive fitness")

    fig.suptitle("4D MAP-Elites Archive — Pairwise Fitness Projections", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = join(output_dir, "archive_pairwise_projections_4d.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved 4D pairwise projection plot to: {save_path}")


def save_selected_bodies_sparse(
    output_dir: str,
    archive: dict[tuple[int, ...], dict[str, Any]],
    fixed_controller_params: np.ndarray,
    n_select: int,
) -> None:
    """Save the top archive elites as body files and full fixed-controller genotypes."""
    selected_dir = join(output_dir, "selected_bodies")
    os.makedirs(selected_dir, exist_ok=True)

    elites = list(archive.items())
    elites.sort(key=lambda item: item[1]["fitness"], reverse=True)
    selected = elites[:n_select]

    print("\nSelected sparse 4D MAP-Elites elites:")
    for idx, (cell, elite) in enumerate(selected):
        body = elite["body"]
        scores = elite["scores"]
        fitness = elite["fitness"]
        descriptor = elite["descriptor"]

        full_genotype = np.concatenate([fixed_controller_params, body])

        np.save(join(selected_dir, f"body_{idx:02d}.npy"), body)
        np.save(join(selected_dir, f"full_genotype_fixed_controller_{idx:02d}.npy"), full_genotype)

        if idx == 0:
            np.save(join(selected_dir, "x_best.npy"), full_genotype)

        print(
            f"{idx:02d} | cell={cell} | fit={fitness:.2f} | "
            f"scores={np.round(scores, 2)} | lengths={np.round(descriptor, 3)}"
        )

    print(f"Saved selected bodies to: {selected_dir}")


# ---------------------------------------------------------------------------
# Main 4D sparse MAP-Elites algorithm
# ---------------------------------------------------------------------------

def run_map_elites_body_search_4d(
    main_module_name: str,
    baseline_dir: str,
    output_dir: str,
    n_bins_per_dim: int = 8,
    n_init: int = 128,
    n_iterations: int = 1000,
    mutation_sigma: float = 0.15,
    n_repeats: int = 2,
    n_steps: int = 300,
    n_select: int = 10,
    fitness_mode: str = "sum",
    seeded_fraction: float = 0.6,
    random_seed: int = 123,
    save_every: int = 100,
) -> None:
    """Run sparse 4D MAP-Elites over body parameters only."""
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    main = import_main_module(main_module_name)
    world = main.FinalWorld()

    baseline_x = load_baseline_genotype(baseline_dir, main)

    if len(baseline_x) != world.n_params:
        raise ValueError(
            f"Baseline genotype length mismatch. Loaded {len(baseline_x)}, "
            f"but {main_module_name}.FinalWorld expects {world.n_params}.\n"
            "This usually means your controller architecture changed. Use a baseline generated "
            "with the same FinalWorld/controller architecture."
        )

    fixed_controller_params = baseline_x[:world.n_weights]
    baseline_body_params = baseline_x[world.n_weights:]

    total_possible_cells = n_bins_per_dim ** 4

    print("\n" + "=" * 88)
    print("Sparse 4D MAP-Elites body search with fixed baseline controller")
    print("=" * 88)
    print(f"Main module        : {main_module_name}")
    print(f"Baseline dir       : {baseline_dir}")
    print(f"Output dir         : {output_dir}")
    print(f"Controller fixed   : {world.n_weights} params")
    print(f"Body searched      : {world.n_body_params} params")
    print(f"Descriptor dims    : 4 total leg lengths (FL, FR, BL, BR)")
    print(f"Bins per dim       : {n_bins_per_dim}")
    print(f"Possible cells     : {total_possible_cells}")
    print(f"Init / iterations  : {n_init} / {n_iterations}")
    print(f"Mutation sigma     : {mutation_sigma}")
    print(f"Repeats / steps    : {n_repeats} / {n_steps}")
    print(f"Fitness mode       : {fitness_mode}")
    print(f"Seeded fraction    : {seeded_fraction}")
    print("=" * 88)

    archive: dict[tuple[int, ...], dict[str, Any]] = {}

    def evaluate_and_insert(body: np.ndarray, label: str) -> None:
        scores = evaluate_body_with_fixed_controller(
            world,
            body,
            fixed_controller_params,
            n_repeats=n_repeats,
            n_steps=n_steps,
        )
        inserted, cell, fitness = try_insert_sparse(
            archive,
            body,
            scores,
            n_bins_per_dim=n_bins_per_dim,
            fitness_mode=fitness_mode,
        )
        print(
            f"{label} | scores={np.round(scores, 2)} | fit={fitness:.2f} | "
            f"cell={cell} | insert={inserted} | occupied={len(archive)}"
        )

    # Insert original baseline body first.
    evaluate_and_insert(baseline_body_params, "Baseline body")

    # Initialization: mix of mutations around baseline body and random bodies.
    n_seeded = int(round(seeded_fraction * n_init))
    n_random = n_init - n_seeded

    print("\nInitialization phase...")
    for k in range(n_seeded):
        body = mutate_body(baseline_body_params, sigma=mutation_sigma, bounds=(-1.0, 1.0))
        evaluate_and_insert(body, f"Seeded init {k + 1:04d}/{n_seeded}")

    for k in range(n_random):
        body = np.random.uniform(-1.0, 1.0, size=world.n_body_params)
        evaluate_and_insert(body, f"Random init {k + 1:04d}/{n_random}")

    print("\nMAP-Elites iteration phase...")
    for it in range(n_iterations):
        if len(archive) == 0:
            parent = np.random.uniform(-1.0, 1.0, size=world.n_body_params)
        else:
            chosen_cell = list(archive.keys())[np.random.randint(len(archive))]
            parent = archive[chosen_cell]["body"]

        child = mutate_body(parent, sigma=mutation_sigma, bounds=(-1.0, 1.0))

        scores = evaluate_body_with_fixed_controller(
            world,
            child,
            fixed_controller_params,
            n_repeats=n_repeats,
            n_steps=n_steps,
        )

        inserted, cell, fitness = try_insert_sparse(
            archive,
            child,
            scores,
            n_bins_per_dim=n_bins_per_dim,
            fitness_mode=fitness_mode,
        )

        if it % 10 == 0 or inserted:
            best_fitness = max(elite["fitness"] for elite in archive.values())
            print(
                f"Iter {it:05d}/{n_iterations} | scores={np.round(scores, 2)} | "
                f"fit={fitness:.2f} | cell={cell} | insert={inserted} | "
                f"occupied={len(archive)} | best={best_fitness:.2f}"
            )

        if save_every > 0 and it > 0 and it % save_every == 0:
            save_sparse_archive(output_dir, archive)
            write_sparse_archive_summary(output_dir, archive, n_bins_per_dim, fitness_mode)
            plot_sparse_archive_projection(output_dir, archive, n_bins_per_dim)
            plot_archive_pairwise_projections_4d(output_dir, archive, n_bins_per_dim)

    save_sparse_archive(output_dir, archive)
    write_sparse_archive_summary(output_dir, archive, n_bins_per_dim, fitness_mode)
    plot_sparse_archive_projection(output_dir, archive, n_bins_per_dim)
    plot_archive_pairwise_projections_4d(output_dir, archive, n_bins_per_dim)
    save_selected_bodies_sparse(output_dir, archive, fixed_controller_params, n_select=n_select)

    print("\nFinished sparse 4D MAP-Elites body search.")


def evaluate_selected_best(main_module_name: str, output_dir: str, n_episodes: int) -> None:
    """Evaluate selected_bodies/x_best.npy using the main evaluate_checkpoint."""
    main = import_main_module(main_module_name)
    selected_dir = join(output_dir, "selected_bodies")

    xbest_path = join(selected_dir, "x_best.npy")
    if not os.path.isfile(xbest_path):
        raise FileNotFoundError(f"Missing {xbest_path}")

    main.evaluate_checkpoint(
        checkpoint_dir=selected_dir,
        output_dir=join(selected_dir, "evaluation_last"),
        n_episodes=n_episodes,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--main_module",
        type=str,
        default="final_project_train_no_map_elites",
        help="Python module containing FinalWorld and evaluate_checkpoint.",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Folder containing baseline x_best.npy. Default: ROOT_DIR/results/nsga_baseline_no_map_elites",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save MAP-Elites outputs. Default: ROOT_DIR/results/map_elites_body_stage1_4d",
    )

    parser.add_argument("--n_bins_per_dim", type=int, default=8)
    parser.add_argument("--n_init", type=int, default=128)
    parser.add_argument("--n_iterations", type=int, default=1000)
    parser.add_argument("--mutation_sigma", type=float, default=0.15)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--n_steps", type=int, default=300)
    parser.add_argument("--n_select", type=int, default=10)
    parser.add_argument("--seeded_fraction", type=float, default=0.6)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument(
        "--fitness_mode",
        type=str,
        choices=["sum", "min", "mixed", "mean_minus_std"],
        default="sum",
    )
    parser.add_argument("--evaluate_after", action="store_true")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=32)

    args = parser.parse_args()

    main_mod = import_main_module(args.main_module)
    root_dir = getattr(main_mod, "ROOT_DIR", os.getcwd())

    baseline_dir = args.baseline_dir
    if baseline_dir is None:
        baseline_dir = join(root_dir, "results", "nsga_baseline_no_map_elites")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = join(root_dir, "results", "map_elites_body_stage1_4d")

    if args.evaluate_only:
        evaluate_selected_best(args.main_module, output_dir, n_episodes=args.eval_episodes)
        return

    run_map_elites_body_search_4d(
        main_module_name=args.main_module,
        baseline_dir=baseline_dir,
        output_dir=output_dir,
        n_bins_per_dim=args.n_bins_per_dim,
        n_init=args.n_init,
        n_iterations=args.n_iterations,
        mutation_sigma=args.mutation_sigma,
        n_repeats=args.n_repeats,
        n_steps=args.n_steps,
        n_select=args.n_select,
        fitness_mode=args.fitness_mode,
        seeded_fraction=args.seeded_fraction,
        random_seed=args.random_seed,
        save_every=args.save_every,
    )

    if args.evaluate_after:
        evaluate_selected_best(args.main_module, output_dir, n_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
