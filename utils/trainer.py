import numpy as np
import pickle
import time
import os 
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
# import wandb 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import json

from utils.optimization_utils import *
from utils.lbfgs import nondiff_lbfgs_solve, hybrid_lbfgs_solve
from models.neural_networks import build_network

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)


def _json_safe(value: Any) -> Any:
    """Convert metric records to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
        return _json_safe(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def _progress_postfix(metrics: Dict[str, float], include_loss: bool = False) -> Dict[str, str]:
    """Select a compact set of metrics for tqdm without hiding the full history."""
    postfix = {}
    if include_loss and 'loss' in metrics:
        postfix['loss'] = f"{metrics['loss']:.3e}"
    objective = metrics.get('objective', metrics.get('obj'))
    if objective is not None:
        postfix['obj'] = f"{objective:.3e}"
    if 'opt_gap_mean' in metrics:
        postfix['gap'] = f"{metrics['opt_gap_mean']:.2e}"
    if 'feasibility_rate' in metrics:
        postfix['feas'] = f"{100 * metrics['feasibility_rate']:.1f}%"
    eq_violation = metrics.get('eq_violation_max_mean', metrics.get('eq_violation_l1'))
    ineq_violation = metrics.get('ineq_violation_max_mean', metrics.get('ineq_violation_l1'))
    if eq_violation is not None:
        postfix['eq'] = f"{eq_violation:.2e}"
    if ineq_violation is not None:
        postfix['ineq'] = f"{ineq_violation:.2e}"
    return postfix


def load_instance(config):
    """Loads problem instance, data, and sets up save directory."""

    # Load data
    seed = config['seed']
    method = config['method']
    val_size = config['val_size']
    test_size = config['test_size']
    prob_type = config['prob_type']
    prob_name = config['prob_name']
    prob_size = config['prob_size']

    # Map problem types to their corresponding problem classes
    if prob_type == 'convex':
        problem_names = {
            'qp': QPProblem,
            'qcqp': QCQPProblem,
            'socp': SOCPProblem,
        }
    elif prob_type == 'nonconvex':
        problem_names = {
            'qp': nonconvexQPProblem,
            'qcqp': nonconvexQCQPProblem,
            'socp': nonconvexSOCPProblem,
        }
    elif prob_type == 'nonsmooth_nonconvex':
        problem_names = {
            'qp': nonsmooth_nonconvexQPProblem,
            'qcqp': nonsmooth_nonconvexQCQPProblem,
            'socp': nonsmooth_nonconvexSOCPProblem,
        }
    
    if prob_name not in problem_names:
        raise NotImplementedError(f"Problem type '{prob_type}_{prob_name}' not implemented")
    
    # Construct filepath using consistent pattern
    seed_data = 2025
    filepath = os.path.join(
        'datasets', 
        prob_type, 
        prob_name,
        f"random{seed_data}_{prob_name}_dataset_var{prob_size[0]}_ineq{prob_size[1]}_eq{prob_size[2]}_ex{prob_size[3]}"
    )
    
    # Load dataset
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    
    # Create problem instance using the appropriate class
    data = problem_names[prob_name](dataset, val_size, test_size, seed)

    data.device = DEVICE
    print("Running on: ", DEVICE)
    for attr in dir(data):
        var = getattr(data, attr)
        if torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass

    if config['ablation'] == True:
        result_save_dir = os.path.join('ablation_results', prob_type, prob_name, str(data), config['network'] + '_' + config['method'], 'dist_'+ str(config['FSNet']['dist_weight']) + '_diff_' + str(config['FSNet']['max_diff_iter']))
    else:
        result_save_dir = os.path.join('results', prob_type, prob_name, str(data), config['network'] + '_' + config['method'])

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    
    return data, result_save_dir


def create_model(data, method, config):
    """Creates and returns a neural network model."""
    
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    network = config['network']
    dropout = config["dropout"]

    if method == "DC3":
        out_dim = data.partial_vars.shape[0]
    else:
        out_dim = data.ydim
    model = build_network(
        network,
        data.xdim,
        hidden_dim,
        out_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    return model.to(DEVICE)


class Trainer:
    def __init__(self, data, config, save_dir=None):
        """Initializes the Trainer with data, method, and configuration."""
        self.data = data
        self.method = config['method']
        self.config = config
        self.save_dir = save_dir
        
        self.config_method = config[self.method]
        self.evaluator = Evaluator(data, self.method, config)
        self.metric_history = []
        self.metrics_jsonl_path = None
        
        self._initialize_params()

    def compute_loss(
        self,
        X_batch: torch.Tensor,
        Y_pred: torch.Tensor,
        Y_true: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the loss and additional metrics."""
        Y_pred_scaled = self.data.scale(Y_pred)
        metrics = {}
        if self.method == "penalty":
            loss, metrics = self._penalty_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "adaptive_penalty":
            loss, metrics = self._adaptive_penalty_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "FSNet":
            loss, metrics = self._fsnet_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "DC3":
            loss, metrics = self._dc3_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "projection":
            loss, metrics = self._projection_loss(X_batch, Y_pred_scaled, metrics)
        else:
            raise ValueError(f"Unknown training method: {self.method}")

        if Y_true is not None:
            metrics.update(
                self.evaluator._compute_batch_metrics(X_batch, metrics.pop('_Y_final'), Y_true)
            )
        else:
            metrics.pop('_Y_final', None)
        return loss, metrics
        

    def _penalty_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the penalty loss."""
        obj = self.data.obj_fn(Y_pred_scaled)
        eq_violation = self.data.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        ineq_violation = self.data.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        eq_violation_l1 = self.data.eq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
        ineq_violation_l1 = self.data.ineq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
    
        loss = self.config_method['obj_weight'] * obj + \
               self.config_method['eq_pen_weight'] * eq_violation + \
               self.config_method['ineq_pen_weight'] * ineq_violation 

        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            '_Y_final': Y_pred_scaled,
        })
        return loss, metrics

    def _adaptive_penalty_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the adaptive penalty loss."""
        obj = self.data.obj_fn(Y_pred_scaled)
        eq_violation = self.data.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        ineq_violation = self.data.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        eq_violation_l1 = self.data.eq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
        ineq_violation_l1 = self.data.ineq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)

        loss = self.config_method['obj_weight'] * obj + \
               self.adaptive_eq_weight * eq_violation + \
               self.adaptive_ineq_weight * ineq_violation

        with torch.no_grad():
            self.adaptive_eq_weight = torch.clamp(self.adaptive_eq_weight + self.config_method['increasing_rate'] * eq_violation.mean(), min=0.0, max=self.config_method['eq_pen_weight_max'])
            self.adtaptive_ineq_weight = torch.clamp(self.adaptive_ineq_weight + self.config_method['increasing_rate'] * ineq_violation.mean(), min=0.0, max=self.config_method['ineq_pen_weight_max'])
            if self.adaptive_eq_weight >= self.config_method['eq_pen_weight_max']:
                self.adaptive_eq_weight = self.config_method['eq_pen_weight_max']/2
            if self.adaptive_ineq_weight >= self.config_method['ineq_pen_weight_max']:
                self.adaptive_ineq_weight = self.config_method['ineq_pen_weight_max']/2

        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            '_Y_final': Y_pred_scaled,
        })
        return loss, metrics
    
    def _fsnet_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the FSNet loss."""
        pre_eq_violation = self.data.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        pre_ineq_violation = self.data.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        Y_final = hybrid_lbfgs_solve(
            X_batch,
            Y_pred_scaled,
            self.data,
            val_tol=self.config_method['val_tol'],
            memory=self.config_method['memory_size'],
            max_iter=self.config_method['max_iter'],
            max_diff_iter=self.config_method['max_diff_iter'],
            scale=self.config_method['scale'],
        )
        obj = self.data.obj_fn(Y_final)
        eq_violation = self.data.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.data.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.data.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.data.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square().mean()

        if pre_eq_violation.mean() >= 1e3 or pre_ineq_violation.mean() >= 1e3:
            loss = self.config_method['obj_weight'] * obj + \
                   self.config_method['dist_weight'] * distance +\
                   self.config_method['eq_pen_weight'] * pre_eq_violation + \
                   self.config_method['ineq_pen_weight'] * pre_ineq_violation
        else:
            loss = self.config_method['obj_weight'] * obj + \
                   self.config_method['dist_weight'] * distance
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.item(),
            '_Y_final': Y_final,
        })
        return loss, metrics
    

    def _dc3_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the DC3 loss."""
        Y_completion = self.data.complete_partial(X_batch, Y_pred_scaled)
        Y_final = grad_steps(self.data, X_batch, Y_completion, self.config)
        obj = self.data.obj_fn(Y_final)
        eq_violation = self.data.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.data.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.data.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.data.ineq_resid(X_batch, Y_final).abs().sum(dim=1)
        
        loss = self.config_method['obj_weight'] * obj + \
               self.config_method['eq_pen_weight'] * eq_violation + \
               self.config_method['ineq_pen_weight'] * ineq_violation
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            '_Y_final': Y_final,
        })

        return loss, metrics
    
    def _projection_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the projection loss."""
        Y_final = self.data.qpth_projection(X_batch, Y_pred_scaled)
        obj = self.data.obj_fn(Y_final)
        eq_violation = self.data.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.data.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.data.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.data.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square().mean()

        loss = self.config_method['obj_weight'] * obj + \
               self.config_method['dist_weight'] * distance
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.item(),
            '_Y_final': Y_final,
        })

        return loss, metrics

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        batch_metrics_history = []
        epoch_start = time.perf_counter()
        learning_rate = self.optimizer.param_groups[0]['lr']
        progress_enabled = self.config.get('progress_bar', True)
        batch_progress = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{self.config['num_epochs']}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not progress_enabled,
        )

        for X_batch, Y_true in batch_progress:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            Y_true = Y_true.to(DEVICE, non_blocking=True)
            Y_pred = self.model(X_batch)

            loss, batch_metrics = self.compute_loss(X_batch, Y_pred, Y_true)

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            batch_metrics['loss'] = loss.mean().item()
            batch_metrics_history.append(batch_metrics)
            running_metrics = self.evaluator._aggregate_metrics(batch_metrics_history)
            batch_progress.set_postfix(_progress_postfix(running_metrics, include_loss=True))

        self.scheduler.step()

        epoch_metrics = self.evaluator._aggregate_metrics(batch_metrics_history)
        epoch_time = time.perf_counter() - epoch_start
        epoch_metrics.update({
            'epoch_time_seconds': epoch_time,
            'samples_per_second': epoch_metrics.get('num_samples', 0) / max(epoch_time, 1e-12),
            'learning_rate': learning_rate,
            'num_batches': len(train_loader),
        })
        return epoch_metrics
    
    def _initialize_params(self) -> None:
        if self.method == 'adaptive_penalty':
            self.adaptive_eq_weight = self.config_method['eq_pen_weight']
            self.adaptive_ineq_weight = self.config_method['ineq_pen_weight']
           
    def _update_epoch_params(self, epoch: int) -> None:
        """Update parameters based on epoch."""
        # FSNet tolerance decay
        if (self.method == 'FSNet' and (epoch + 1) % self.config_method['decay_tol_step'] == 0):
            self.config_method['val_tol'] = np.clip(
                self.config_method['val_tol'] / 10, 
                a_min=1e-9, 
                a_max=1e-6
            )
        
        # Dropout decay
        if epoch == 100:
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = m.p / 2
        elif epoch == 150:
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0

    def _start_metric_logging(self) -> None:
        """Initialize in-memory history and the interruption-safe JSONL log."""
        self.metric_history = []
        if not self.save_dir:
            return

        os.makedirs(self.save_dir, exist_ok=True)
        filename = f"metrics_seed{self.config.get('seed', 'N_A')}.jsonl"
        self.metrics_jsonl_path = os.path.join(self.save_dir, filename)
        with open(self.metrics_jsonl_path, 'w', encoding='utf-8'):
            pass

    def _record_metrics(self, split: str, metrics: Dict[str, Any], **context: Any) -> Dict[str, Any]:
        """Record one timestamped metric snapshot in memory and on disk."""
        record = _json_safe({
            'split': split,
            'timestamp': datetime.now().astimezone().isoformat(timespec='seconds'),
            'seed': self.config.get('seed'),
            'method': self.method,
            'problem_type': self.config.get('prob_type'),
            'problem_name': self.config.get('prob_name'),
            'feasibility_tolerance': self.evaluator.feasibility_tol,
            **context,
            **metrics,
        })
        self.metric_history.append(record)
        if self.metrics_jsonl_path:
            with open(self.metrics_jsonl_path, 'a', encoding='utf-8') as file:
                file.write(json.dumps(record, allow_nan=False) + '\n')
        return record
 
    def train(self):
        """Main training loop with detailed results collection."""
        train_loader = DataLoader(
            self.data.train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
        )
        
        val_loader = DataLoader(
            self.data.val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Initialize model
        self.model = create_model(self.data, self.method, self.config)
        
        # Initialize optimizer and scheduler (fix the initialization issue)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=0.001, 
            fused=DEVICE.type == 'cuda'
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config['lr_decay_step'], 
            gamma=self.config['lr_decay']
        )

        self._start_metric_logging()

        # Training history
        train_history = []
        val_history = []

        train_start = time.perf_counter()
        progress_enabled = self.config.get('progress_bar', True)
        epoch_progress = tqdm(
            range(self.config['num_epochs']),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
            disable=not progress_enabled,
        )
        for epoch in epoch_progress:
            self._update_epoch_params(epoch)

            # Train for one epoch
            epoch_metrics = self.train_epoch(train_loader, epoch)
            elapsed_time = time.perf_counter() - train_start
            train_record = self._record_metrics(
                'train',
                epoch_metrics,
                epoch=epoch,
                epoch_number=epoch + 1,
                elapsed_time_seconds=elapsed_time,
            )
            train_history.append(train_record)
            epoch_postfix = _progress_postfix(epoch_metrics, include_loss=True)
            epoch_postfix['time'] = f"{elapsed_time:.1f}s"
            epoch_progress.set_postfix(epoch_postfix)

            # Evaluate on validation set
            if (epoch + 1) % self.config['eval_step'] == 0:
                val_metrics = self.evaluator.evaluate(self.model, val_loader, f"validation_epoch_{epoch+1}")
                val_record = self._record_metrics(
                    'validation',
                    val_metrics,
                    epoch=epoch,
                    epoch_number=epoch + 1,
                    elapsed_time_seconds=time.perf_counter() - train_start,
                )
                val_history.append(val_record)

        training_time = time.perf_counter() - train_start
        print(f"\nTraining completed in {training_time:.2f} seconds.")

        # Enhanced test evaluation with multiple batch sizes and detailed results
        if hasattr(self.data, 'test_dataset'):
            print("\n" + "="*60)
            print("COMPREHENSIVE TEST EVALUATION WITH DETAILED RESULTS")
            print("="*60)
            
            # Get test batch sizes from config or use defaults
            test_batch_sizes = self.config.get(
                'test_batch_sizes', self.config.get('test_batch_size', [256, 512])
            )
            
            print(f"Testing with batch sizes: {test_batch_sizes}")
            
            # Run evaluation with all batch sizes and collect detailed results for all
            batch_size_results, all_detailed_results = self.evaluator.evaluate_multiple_batch_sizes(
                self.model, 
                self.data.test_dataset, 
                test_batch_sizes, 
                "test"
            )
            
            # Combine all test results
            final_test_results = {
                'batch_size_comparison': batch_size_results,
                'detailed_results_all_batch_sizes': all_detailed_results
            }
            for batch_size, result in batch_size_results.items():
                if 'metrics' in result:
                    self._record_metrics(
                        'test',
                        result['metrics'],
                        batch_size=int(batch_size),
                        elapsed_time_seconds=time.perf_counter() - train_start,
                    )
        else:
            print("No test dataset available")
            final_test_results = {}
            all_detailed_results = None
        
        # Save all results with detailed information
        if self.save_dir:
            self._save_model_and_results(
                train_history, 
                val_history, 
                final_test_results, 
                training_time
            )
        
        return self.model
    
    
    def _save_model_and_results(self, train_history, val_history,
                                test_results_data, training_time):
        """Saves the model in a .pt file and other results in a .pkl file."""
        if not self.save_dir:
            print("Save directory not specified. Skipping saving.")
            return
        
        os.makedirs(self.save_dir, exist_ok=True) # Ensure save directory exists
        print(f"\nSaving model and results to: {self.save_dir}")

        # --- 1. Save Model File (.pt) ---
        model_save_content = {
            'model_state_dict': self.model.state_dict(),
            'model_architecture_str': str(self.model), 
            'config': self.config, # Include config for easier model reloading
        }
        model_filename = f"model_seed{self.config.get('seed', 'N_A')}.pt"
        model_filepath = os.path.join(self.save_dir, model_filename)
        try:
            torch.save(model_save_content, model_filepath)
            print(f"✓ Model saved: {model_filepath}")
        except Exception as e:
            print(f"✗ Error saving model: {e}")


        # --- 2. Save Results File (.pkl) ---
        results_save_content = {
            'seed': self.config.get('seed', 'N_A'),
            'method': self.method,
            'config': self.config, # Full config for reference
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'training_time_seconds': training_time,
            'train_history': train_history,
            'val_history': val_history,
            'metric_history': self.metric_history,
            'test_results': test_results_data, # This contains summary and detailed results
            'pytorch_version': torch.__version__,
            'device_used': str(DEVICE)
        }

        results_filename = f"results_seed{self.config.get('seed', 'N_A')}.pkl"
        results_filepath = os.path.join(self.save_dir, results_filename)
        try:
            with open(results_filepath, 'wb') as f:
                pickle.dump(results_save_content, f)
            print(f"✓ Detailed results saved: {results_filepath}")
        except Exception as e:
            print(f"✗ Error saving results: {e}")

        # --- 3. Save portable metrics history (.json) ---
        metrics_filename = f"metrics_seed{self.config.get('seed', 'N_A')}.json"
        metrics_filepath = os.path.join(self.save_dir, metrics_filename)
        metrics_save_content = _json_safe({
            'seed': self.config.get('seed', 'N_A'),
            'method': self.method,
            'problem_type': self.config.get('prob_type'),
            'problem_name': self.config.get('prob_name'),
            'feasibility_tolerance': self.evaluator.feasibility_tol,
            'optimality_gap_epsilon': self.evaluator.optimality_gap_epsilon,
            'device_used': str(DEVICE),
            'records': self.metric_history,
        })
        try:
            with open(metrics_filepath, 'w', encoding='utf-8') as file:
                json.dump(metrics_save_content, file, indent=2, allow_nan=False)
            print(f"Metrics history saved: {metrics_filepath}")
        except Exception as e:
            print(f"Error saving metrics history: {e}")

        print(f"\nFiles saved (or attempted):")
        print(f"  - {model_filename} (model weights and architecture)")
        print(f"  - {results_filename} (training history, metrics, detailed test results)")
        print(f"  - {metrics_filename} (portable metric history)")
        if self.metrics_jsonl_path:
            print(f"  - {os.path.basename(self.metrics_jsonl_path)} (incremental metric history)")




class Evaluator:
    """Separate evaluator class for model evaluation."""
    
    def __init__(self, data, method, config):
        """Initialize evaluator."""
        self.data = data
        self.method = method
        self.config = config
        self.config_method = config[method]
        self.feasibility_tol = float(config.get('feasibility_tol', 1e-5))
        self.optimality_gap_epsilon = float(config.get('optimality_gap_epsilon', 1e-8))
    
    @torch.no_grad()
    def evaluate(self, model, data_loader, split_name="eval", return_detailed=False):
        """
        Comprehensive evaluation of the model.
        
        Args:
            model: The neural network model
            data_loader: DataLoader for evaluation data
            split_name: Name of the split (train/val/test)
            return_detailed: Whether to return detailed predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_metrics = []
        detailed_results = [] if return_detailed else None
        
        total_time = 0.0
        progress_enabled = self.config.get('progress_bar', True)
        evaluation_progress = tqdm(
            data_loader,
            desc=split_name,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not progress_enabled,
        )

        for X_batch, Y_true in evaluation_progress:
            X_batch = X_batch.to(DEVICE)
            Y_true = Y_true.to(DEVICE)

            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Forward pass
            Y_pred = model(X_batch)
            Y_pred_scaled = self.data.scale(Y_pred)
            
            # Method-specific post-processing
            Y_final = self._post_process_predictions(X_batch, Y_pred_scaled)

            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            batch_time = time.perf_counter() - start_time
            total_time += batch_time
            
            # Compute comprehensive metrics
            batch_metrics = self._compute_batch_metrics(X_batch, Y_final, Y_true)
            batch_metrics['inference_time'] = batch_time
            all_metrics.append(batch_metrics)
            running_metrics = self._aggregate_metrics(all_metrics)
            running_metrics['elapsed_time_seconds'] = total_time
            evaluation_progress.set_postfix(_progress_postfix(running_metrics))
            
            # Store detailed results if requested
            if return_detailed:
                detailed_results.append({
                    'X': X_batch.cpu(),
                    'Y_pred': Y_pred.cpu(),
                    'Y_pred_scaled': Y_pred_scaled.cpu(),
                    'Y_final': Y_final.cpu(),
                    'Y_true': Y_true.cpu(),
                    'metrics': batch_metrics
                })
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        aggregated_metrics['total_time'] = total_time
        aggregated_metrics['avg_inference_time'] = total_time / max(len(data_loader), 1)
        aggregated_metrics['avg_inference_time_per_sample'] = (
            total_time / max(aggregated_metrics.get('num_samples', 0), 1)
        )
        aggregated_metrics['samples_per_second'] = (
            aggregated_metrics.get('num_samples', 0) / max(total_time, 1e-12)
        )
        aggregated_metrics['num_batches'] = len(data_loader)
        
        # Print summary
        self._print_evaluation_summary(split_name, aggregated_metrics)
        
        if return_detailed:
            return aggregated_metrics, detailed_results
        return aggregated_metrics
    
    @torch.enable_grad()
    def _post_process_predictions(self, X_batch, Y_pred_scaled):
        """Apply method-specific post-processing."""
        if self.method in ["penalty", "adaptive_penalty"]:
            return Y_pred_scaled
        elif self.method == "FSNet":
            return nondiff_lbfgs_solve(
                X_batch, Y_pred_scaled, self.data,
                val_tol=self.config_method.get('test_val_tol', 1e-6),
                memory=self.config_method.get('memory_size', 20),
                max_iter=self.config_method.get('max_iter', 20),
                scale=self.config_method.get('scale', 1)
            )
        elif self.method == "DC3":
            Y_completion = self.data.complete_partial(X_batch, Y_pred_scaled)
            return grad_steps(self.data, X_batch, Y_completion, self.config)
        elif self.method == "projection":
            return self.data.qpth_projection(X_batch, Y_pred_scaled)
        else:
            return Y_pred_scaled
    
    def _compute_batch_metrics(self, X_batch, Y_final, Y_true):
        """Compute comprehensive metrics for a batch."""
        # Objective values
        obj_pred = self.data.obj_fn(Y_final)
        obj_true = self.data.obj_fn(Y_true)
        
        # Constraint violations
        eq_resid = self.data.eq_resid(X_batch, Y_final)
        ineq_resid = self.data.ineq_resid(X_batch, Y_final)
        
        eq_violation_l2 = eq_resid.square().sum(dim=1)
        ineq_violation_l2 = ineq_resid.square().sum(dim=1)
        eq_violation_l1 = eq_resid.abs().sum(dim=1)
        ineq_violation_l1 = ineq_resid.abs().sum(dim=1)
        eq_violation_max = eq_resid.abs().max(dim=1)[0]
        ineq_violation_max = ineq_resid.abs().max(dim=1)[0]
        
        # Optimality gap. The clamped denominator avoids inf/nan near zero.
        objective_gap = obj_pred - obj_true
        denominator = obj_true.abs().clamp_min(
            max(self.optimality_gap_epsilon, torch.finfo(obj_true.dtype).eps)
        )
        opt_gap = objective_gap / denominator
        absolute_opt_gap = opt_gap.abs()
        absolute_objective_gap = objective_gap.abs()
        # Solution distance
        solution_distance = torch.norm(Y_final - Y_true, dim=1).square()

        eq_feasible = eq_violation_max <= self.feasibility_tol
        ineq_feasible = ineq_violation_max <= self.feasibility_tol
        feasible = eq_feasible & ineq_feasible
        
        return {
            # Objective metrics
            'objective': obj_pred.mean().item(),
            'true_objective': obj_true.mean().item(),
            'objective_gap_mean': objective_gap.mean().item(),
            'objective_gap_std': objective_gap.std(unbiased=False).item(),
            'objective_gap_max': objective_gap.max().item(),
            'objective_gap_min': objective_gap.min().item(),
            'absolute_objective_gap_mean': absolute_objective_gap.mean().item(),
            'absolute_objective_gap_std': absolute_objective_gap.std(unbiased=False).item(),
            'absolute_objective_gap_max': absolute_objective_gap.max().item(),
            'opt_gap_mean': opt_gap.mean().item(),
            'opt_gap_std': opt_gap.std(unbiased=False).item(),
            'opt_gap_max': opt_gap.max().item(),
            'opt_gap_min': opt_gap.min().item(),
            'absolute_opt_gap_mean': absolute_opt_gap.mean().item(),
            'absolute_opt_gap_std': absolute_opt_gap.std(unbiased=False).item(),
            'absolute_opt_gap_max': absolute_opt_gap.max().item(),

            # Feasibility rates at the configured tolerance
            'feasibility_rate': feasible.double().mean().item(),
            'eq_feasibility_rate': eq_feasible.double().mean().item(),
            'ineq_feasibility_rate': ineq_feasible.double().mean().item(),
            
            # Constraint violations (L2)
            'eq_violation_l2_mean': eq_violation_l2.mean().item(),
            'eq_violation_l2_max': eq_violation_l2.max().item(),
            'ineq_violation_l2_mean': ineq_violation_l2.mean().item(),
            'ineq_violation_l2_max': ineq_violation_l2.max().item(),
            
            # Constraint violations (l1)
            'eq_violation_l1_mean': eq_violation_l1.mean().item(),
            'eq_violation_l1_max': eq_violation_l1.max().item(),
            'ineq_violation_l1_mean': ineq_violation_l1.mean().item(),
            'ineq_violation_l1_max': ineq_violation_l1.max().item(),
            
            # Constraint violations (L∞)
            'eq_violation_max_mean': eq_violation_max.mean().item(),
            'eq_violation_max_max': eq_violation_max.max().item(),
            'ineq_violation_max_mean': ineq_violation_max.mean().item(),
            'ineq_violation_max_max': ineq_violation_max.max().item(),
            
            # Solution quality
            'solution_distance_mean': solution_distance.mean().item(),
            'solution_distance_std': solution_distance.std(unbiased=False).item(),
            'solution_distance_max': solution_distance.max().item(),
            'num_samples': X_batch.shape[0],
        }
    
    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across batches."""
        if not all_metrics:
            return {}
        
        keys = set().union(*(metrics.keys() for metrics in all_metrics)) - {'inference_time'}
        aggregated = {}
        total_samples = sum(metrics.get('num_samples', 1) for metrics in all_metrics)

        for key in keys:
            if key == 'num_samples':
                aggregated[key] = total_samples
                continue

            metrics_with_key = [metrics for metrics in all_metrics if key in metrics]
            values = [metrics[key] for metrics in metrics_with_key]
            if key.endswith('_std'):
                mean_key = key.replace('_std', '_mean')
                weighted_count = sum(metrics.get('num_samples', 1) for metrics in metrics_with_key)
                mean = sum(
                    metrics.get('num_samples', 1) * metrics[mean_key]
                    for metrics in metrics_with_key
                ) / max(weighted_count, 1)
                second_moment = sum(
                    metrics.get('num_samples', 1)
                    * (metrics[key] ** 2 + metrics[mean_key] ** 2)
                    for metrics in metrics_with_key
                ) / max(weighted_count, 1)
                aggregated[key] = float(np.sqrt(max(second_moment - mean ** 2, 0.0)))
            elif key.endswith('_max'):
                aggregated[key] = max(values)
            elif key.endswith('_min'):
                aggregated[key] = min(values)
            else:
                weighted_count = sum(metrics.get('num_samples', 1) for metrics in metrics_with_key)
                aggregated[key] = sum(
                    metrics.get('num_samples', 1) * metrics[key]
                    for metrics in metrics_with_key
                ) / max(weighted_count, 1)
        
        return aggregated
    
    def _print_evaluation_summary(self, split_name, metrics):
        """Print evaluation summary."""
        print(f"\n{split_name.upper()} EVALUATION RESULTS:")
        print("=" * 50)
        print(f"Objective Value:     {metrics.get('objective', 0):.6e}")
        print(f"True Objective:      {metrics.get('true_objective', 0):.6e}")
        print(f"Objective Gap:       {metrics.get('objective_gap_mean', 0):.6e} ± {metrics.get('objective_gap_std', 0):.6e}")
        print(f"Relative Opt Gap:    {metrics.get('opt_gap_mean', 0):.6e} ± {metrics.get('opt_gap_std', 0):.6e}")
        print(f"Feasibility Rate:    {100 * metrics.get('feasibility_rate', 0):.2f}% (tol={self.feasibility_tol:.1e})")
        print(f"Eq Feasibility:      {100 * metrics.get('eq_feasibility_rate', 0):.2f}%")
        print(f"Ineq Feasibility:    {100 * metrics.get('ineq_feasibility_rate', 0):.2f}%")
        print(f"Eq Violation l1:   {metrics.get('eq_violation_l1_mean', 0):.6e} (max: {metrics.get('eq_violation_l1_max', 0):.6e})")
        print(f"Ineq Violation l1: {metrics.get('ineq_violation_l1_mean', 0):.6e} (max: {metrics.get('ineq_violation_l1_max', 0):.6e})")
        print(f"Solution Distance:   {metrics.get('solution_distance_mean', 0):.6e} ± {metrics.get('solution_distance_std', 0):.6e}")
        print(f"Avg Inference Time:  {metrics.get('avg_inference_time', 0):.4f}s")
        print(f"Throughput:          {metrics.get('samples_per_second', 0):.2f} samples/s")
        print("=" * 50)
    
    def evaluate_multiple_batch_sizes(self, model, dataset, batch_sizes, split_name="test"):
        """
        Evaluate model with multiple batch sizes and collect detailed results for all successful ones.
        
        Args:
            model: The neural network model
            dataset: Dataset to evaluate on
            batch_sizes: List of batch sizes to test
            split_name: Name of the evaluation split
            
        Returns:
            Tuple of (results_dict, detailed_results_dict)
        """
        results = {}
        all_detailed_results = {}
        
        for batch_size in batch_sizes:
            print(f"\nEvaluating with batch size: {batch_size} (with detailed results)")
            
            try:
                # Create data loader with specific batch size
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                )
                
                # Evaluate with detailed results
                metrics, detailed_results = self.evaluate(
                    model, data_loader, f"{split_name}_bs{batch_size}", 
                    return_detailed=True
                )
                
                results[batch_size] = {
                    'metrics': metrics,
                    'batch_size': batch_size,
                }
                
                all_detailed_results[batch_size] = detailed_results
                
                # Clear cache after each evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Batch size {batch_size} failed due to memory constraints")
                    results[batch_size] = {
                        'error': 'OOM',
                        'batch_size': batch_size
                    }
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        # Print comparison summary
        self._print_batch_size_comparison(results, split_name)
        
        return results, all_detailed_results
    
    def _print_batch_size_comparison(self, results, split_name):
        """Print comparison of results across batch sizes."""
        print(f"\n{split_name.upper()} BATCH SIZE COMPARISON:")
        print("=" * 92)
        print(f"{'Batch Size':<12} {'Objective':<12} {'Opt Gap':<12} {'Feas (%)':<10} {'Eq Viol':<12} {'Ineq Viol':<12} {'Time (s)':<10}")
        print("-" * 92)
        
        for batch_size, result in results.items():
            if 'error' in result:
                print(f"{batch_size:<12} {'OOM':<12} {'OOM':<12} {'OOM':<10} {'OOM':<12} {'OOM':<12} {'OOM':<10}")
            else:
                metrics = result['metrics']
                print(f"{batch_size:<12} "
                      f"{metrics.get('objective', 0):<12.4e} "
                      f"{metrics.get('opt_gap_mean', 0):<12.4e} "
                      f"{100 * metrics.get('feasibility_rate', 0):<10.2f} "
                      f"{metrics.get('eq_violation_l1_mean', 0):<12.4e} "
                      f"{metrics.get('ineq_violation_l1_mean', 0):<12.4e} "
                      f"{metrics.get('total_time', 0):<10.2f}")
        
        print("=" * 92)
        
