import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from compute_weight_to_merit_landscape_random import (
    ARCHITECTURES,
    DEFAULT_DIRECTION_SEEDS,
    HIDDEN_DIM,
    NUM_LAYERS,
    compute_merit_surface_2d,
    create_parser,
    merit_values,
    plot_surfaces,
)
from models.neural_networks import ICNN, MLP, NonNegativeLinear, ResidualMLP


class ArchitectureAblationTest(unittest.TestCase):
    def test_all_backbones_have_four_width_64_hidden_layers(self):
        input_dim, output_dim = 7, 5
        models = {
            "MLP": MLP(input_dim, HIDDEN_DIM, output_dim, num_layers=NUM_LAYERS),
            "ICNN": ICNN(input_dim, HIDDEN_DIM, output_dim, num_layers=NUM_LAYERS),
            "ResMLP": ResidualMLP(
                input_dim, HIDDEN_DIM, output_dim, num_layers=NUM_LAYERS
            ),
        }

        mlp_hidden = [
            layer
            for layer in models["MLP"].mlp[:-2]
            if isinstance(layer, torch.nn.Linear)
        ]
        self.assertEqual(len(mlp_hidden), NUM_LAYERS)
        self.assertTrue(all(layer.out_features == HIDDEN_DIM for layer in mlp_hidden))
        self.assertEqual(len(models["ICNN"].input_layers), NUM_LAYERS)
        self.assertTrue(
            all(layer.out_features == HIDDEN_DIM for layer in models["ICNN"].input_layers)
        )
        self.assertEqual(1 + len(models["ResMLP"].residual_layers), NUM_LAYERS)

        inputs = torch.randn(3, input_dim)
        for model in models.values():
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (3, output_dim))
            self.assertTrue(torch.all(outputs >= 0))
            self.assertTrue(torch.all(outputs <= 1))

    def test_icnn_effective_hidden_weights_are_nonnegative(self):
        model = ICNN(4, HIDDEN_DIM, 3, num_layers=NUM_LAYERS)
        nonnegative_layers = [
            layer for layer in model.modules() if isinstance(layer, NonNegativeLinear)
        ]
        self.assertGreater(len(nonnegative_layers), 0)
        for layer in nonnegative_layers:
            self.assertTrue(torch.all(layer.weight >= 0))

    def test_cli_defaults_cover_three_models_and_five_matched_seeds(self):
        args = create_parser().parse_args(["all"])
        self.assertEqual(tuple(args.architectures), ARCHITECTURES)
        self.assertEqual(tuple(args.direction_seeds), DEFAULT_DIRECTION_SEEDS)
        self.assertEqual(args.penalty_weight, 1e4)
        self.assertEqual(args.grid_size, 31)
        self.assertEqual(args.dpi, 300)


class LandscapeComputationTest(unittest.TestCase):
    def test_merit_uses_requested_l1_penalty_weights(self):
        class ToyProblem:
            def obj_fn(self, prediction):
                return prediction[:, 0]

            def eq_resid(self, inputs, prediction):
                return prediction[:, :1] - inputs[:, :1]

            def ineq_resid(self, inputs, prediction):
                return prediction[:, 1:2] - inputs[:, 1:2]

        inputs = torch.zeros(1, 2)
        prediction = torch.tensor([[2.0, -3.0]])
        merit = merit_values(ToyProblem(), inputs, prediction, 1e4, 1e4)
        self.assertEqual(merit.item(), 2.0 + 1e4 * 2.0 + 1e4 * 3.0)

    def test_surface_is_reproducible_and_restores_model(self):
        torch.manual_seed(11)
        model = MLP(2, 4, 1, num_layers=2).double()
        inputs = torch.zeros(1, 2, dtype=torch.double)
        loader = [(inputs, torch.zeros(1, 1, dtype=torch.double))]

        def parameter_merit(candidate, _loader):
            return float(sum(p.square().sum().item() for p in candidate.parameters()))

        original = [parameter.detach().clone() for parameter in model.parameters()]
        first = compute_merit_surface_2d(
            model,
            loader,
            parameter_merit,
            radius=0.25,
            grid_size=3,
            direction_seed=5,
            norm="filter",
            verbose=False,
        )
        second = compute_merit_surface_2d(
            model,
            loader,
            parameter_merit,
            radius=0.25,
            grid_size=3,
            direction_seed=5,
            norm="filter",
            verbose=False,
        )

        np.testing.assert_allclose(first[2], second[2])
        for parameter, expected in zip(model.parameters(), original):
            torch.testing.assert_close(parameter, expected)

    def test_comparable_plot_files_are_created(self):
        coordinates = np.linspace(-1.0, 1.0, 3)
        x_grid, y_grid = np.meshgrid(coordinates, coordinates)
        architectures = ["MLP", "ICNN", "ResMLP"]
        seeds = [0, 1]
        surfaces = {}
        for architecture_index, architecture in enumerate(architectures):
            for seed in seeds:
                surfaces[(architecture, seed)] = {
                    "X": x_grid,
                    "Y": y_grid,
                    "Z": x_grid**2 + y_grid**2 + architecture_index + seed,
                }

        args = Namespace(
            architectures=architectures,
            direction_seeds=seeds,
            penalty_weight=1e4,
            training_seed=2025,
            dpi=40,
            elevation=30.0,
            azimuth=-60.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            metadata = plot_surfaces(surfaces, args, output_dir)
            self.assertTrue((output_dir / "merit_landscape_all_comparisons.png").exists())
            for seed in seeds:
                self.assertTrue(
                    (output_dir / f"merit_landscape_comparison_direction_seed{seed}.png").exists()
                )
            self.assertEqual(metadata["global_merit_minimum"], 0.0)


if __name__ == "__main__":
    unittest.main()
