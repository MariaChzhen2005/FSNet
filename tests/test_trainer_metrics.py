import json
import math
import tempfile
import unittest

import torch

from utils.trainer import Evaluator, Trainer


class ToyProblem:
    def obj_fn(self, y):
        return y[:, 0] + 2.0

    def eq_resid(self, x, y):
        return y[:, :1] - x[:, :1]

    def ineq_resid(self, x, y):
        return torch.clamp(y[:, 1:2] - 1.0, min=0.0)


class ZeroObjectiveProblem(ToyProblem):
    def obj_fn(self, y):
        return y[:, 0]


def make_config():
    return {
        'method': 'penalty',
        'penalty': {
            'obj_weight': 1.0,
            'eq_pen_weight': 1.0,
            'ineq_pen_weight': 1.0,
        },
        'feasibility_tol': 1e-5,
        'seed': 7,
    }


class EvaluatorMetricsTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator(ToyProblem(), 'penalty', make_config())
        self.x = torch.zeros(3, 2)
        self.y_true = torch.zeros(3, 2)
        self.y_pred = torch.tensor([
            [0.0, 0.0],
            [1e-6, 2.0],
            [1e-3, 0.0],
        ])

    def test_feasibility_rates_and_gap_metrics(self):
        metrics = self.evaluator._compute_batch_metrics(
            self.x, self.y_pred, self.y_true
        )

        self.assertAlmostEqual(metrics['feasibility_rate'], 1 / 3)
        self.assertAlmostEqual(metrics['eq_feasibility_rate'], 2 / 3)
        self.assertAlmostEqual(metrics['ineq_feasibility_rate'], 2 / 3)
        self.assertTrue(math.isfinite(metrics['opt_gap_mean']))
        self.assertAlmostEqual(metrics['opt_gap_max'], 5e-4)
        self.assertEqual(metrics['num_samples'], 3)

    def test_uneven_batches_match_full_batch_aggregation(self):
        full = self.evaluator._compute_batch_metrics(
            self.x, self.y_pred, self.y_true
        )
        batches = [
            self.evaluator._compute_batch_metrics(
                self.x[:2], self.y_pred[:2], self.y_true[:2]
            ),
            self.evaluator._compute_batch_metrics(
                self.x[2:], self.y_pred[2:], self.y_true[2:]
            ),
        ]
        aggregated = self.evaluator._aggregate_metrics(batches)

        for key, expected in full.items():
            if isinstance(expected, (int, float)):
                self.assertAlmostEqual(aggregated[key], expected, places=12, msg=key)

    def test_zero_reference_objective_has_finite_gap(self):
        evaluator = Evaluator(ZeroObjectiveProblem(), 'penalty', make_config())
        metrics = evaluator._compute_batch_metrics(
            torch.zeros(1, 2), torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2)
        )

        self.assertTrue(math.isfinite(metrics['opt_gap_mean']))
        self.assertTrue(math.isfinite(metrics['absolute_opt_gap_mean']))


class MetricLoggingTest(unittest.TestCase):
    def test_jsonl_logging_replaces_non_finite_values_with_null(self):
        with tempfile.TemporaryDirectory() as save_dir:
            trainer = Trainer(ToyProblem(), make_config(), save_dir=save_dir)
            trainer._start_metric_logging()
            trainer._record_metrics('train', {'loss': float('nan')}, epoch=1)

            with open(trainer.metrics_jsonl_path, encoding='utf-8') as file:
                record = json.loads(file.readline())

            self.assertEqual(record['split'], 'train')
            self.assertEqual(record['epoch'], 1)
            self.assertIsNone(record['loss'])
            self.assertIn('timestamp', record)


if __name__ == '__main__':
    unittest.main()
