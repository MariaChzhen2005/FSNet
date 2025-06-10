import numpy as np
import pickle
import time
import os 
from typing import Dict, Tuple
# import wandb 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import json

from utils.optimization_utils import *
from utils.lbfgs import nondiff_lbfgs_solve, hybrid_lbfgs_solve
from models.neural_networks import MLP

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)


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

    if network == 'MLP':
        if method == "DC3":
            out_dim = data.partial_vars.shape[0]
            model = MLP(data.xdim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout)
        else:
            model = MLP(data.xdim, hidden_dim, data.ydim, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model}")
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
        
        self._initialize_params()

    def compute_loss(self, X_batch: torch.Tensor, Y_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the loss and additional metrics."""
        Y_pred_scaled = self.data.scale(Y_pred)
        metrics = {}
        if self.method == "penalty":
            return self._penalty_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "adaptive_penalty":
            return self._adaptive_penalty_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "FSNet":
            return self._fsnet_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "DC3":
            return self._dc3_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "projection":
            return self._projection_loss(X_batch, Y_pred_scaled, metrics)
        

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
        })

        return loss, metrics

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {'obj':0.0, 'loss': 0.0, 'objective': 0.0, 'eq_violation': 0.0, 'ineq_violation': 0.0, 'eq_violation_l1': 0.0, 'ineq_violation_l1': 0.0, 'distance': 0.0}
        
        # Update method parameters if needed
        self._update_epoch_params(epoch)
        
        for batch_idx, (X_batch, _) in enumerate(train_loader):
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            Y_pred = self.model(X_batch)
            
            loss, batch_metrics = self.compute_loss(X_batch, Y_pred)
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
            epoch_metrics['loss'] += loss.mean().item()
        
        self.scheduler.step()
        
        # Average metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def _initialize_params(self) -> None:
        if self.method == 'adaptive_penalty':
            self.adaptive_eq_weight = self.config_method['eq_pen_weight']
            self.adaptive_ineq_weight = self.config_method['ineq_pen_weight']
           
    def _update_epoch_params(self, epoch: int) -> None:
        """Update parameters based on epoch."""
        # FSNet tolerance decay
        if (self.method == 'FSNet' and 
            (epoch + 1) % self.config[self.method]['decay_tol_step'] == 0):
            self.method_params['val_tol'] = np.clip(
                self.method_params['val_tol'] / 10, 
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
            fused=True
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config['lr_decay_step'], 
            gamma=self.config['lr_decay']
        )
        
        # Training history
        train_history = []
        val_history = []

        train_start = time.time()
        for epoch in range(self.config['num_epochs']):
            self._update_epoch_params(epoch)
            epoch_start = time.time()
            
            # Train for one epoch
            self.model.train()
            epoch_metrics = self.train_epoch(train_loader, epoch)
            train_history.append({'epoch': epoch, **epoch_metrics})
            epoch_end = time.time()
       
            # Log metrics
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}, "
                  f"Loss: {epoch_metrics['loss']:.4f}, "
                  f"Obj: {epoch_metrics.get('obj', 0):.4f}, "
                  f"Eq Viol (l1): {epoch_metrics.get('eq_violation_l1', 0):.6f}, "
                  f"Ineq Viol (l1): {epoch_metrics.get('ineq_violation_l1', 0):.6f}, "
                  f"Epoch time: {epoch_end - epoch_start:.2f}s")

            # Evaluate on validation set
            if (epoch + 1) % self.config['eval_step'] == 0:
                print(f"\nRunning validation at epoch {epoch + 1}...")
                val_metrics = self.evaluator.evaluate(self.model, val_loader, f"validation_epoch_{epoch+1}")
                val_history.append({**val_metrics, 'epoch': epoch})
        
        train_end = time.time()
        training_time = train_end - train_start
        print(f"\nTraining completed in {training_time:.2f} seconds.")

        # Enhanced test evaluation with multiple batch sizes and detailed results
        if hasattr(self.data, 'test_dataset'):
            print("\n" + "="*60)
            print("COMPREHENSIVE TEST EVALUATION WITH DETAILED RESULTS")
            print("="*60)
            
            # Get test batch sizes from config or use defaults
            test_batch_sizes = self.config.get('test_batch_sizes', [256, 512])
            
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

        print(f"\nFiles saved (or attempted):")
        print(f"  - {model_filename} (model weights and architecture)")
        print(f"  - {results_filename} (training history, metrics, detailed test results)")




class Evaluator:
    """Separate evaluator class for model evaluation."""
    
    def __init__(self, data, method, config):
        """Initialize evaluator."""
        self.data = data
        self.method = method
        self.config = config
        self.config_method = config[method]
    
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
        
        total_time = 0
        
        for batch_idx, (X_batch, Y_true) in enumerate(data_loader):
            X_batch = X_batch.to(DEVICE)
            Y_true = Y_true.to(DEVICE)
            
            start_time = time.time()
            
            # Forward pass
            Y_pred = model(X_batch)
            Y_pred_scaled = self.data.scale(Y_pred)
            
            # Method-specific post-processing
            Y_final = self._post_process_predictions(X_batch, Y_pred_scaled)
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Compute comprehensive metrics
            batch_metrics = self._compute_batch_metrics(X_batch, Y_final, Y_true)
            batch_metrics['inference_time'] = batch_time
            all_metrics.append(batch_metrics)
            
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
        aggregated_metrics['avg_inference_time'] = total_time / len(data_loader)
        
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
        
        # Optimality gap
        opt_gap = (obj_pred - obj_true) / obj_true.abs()         
        # Solution distance
        solution_distance = torch.norm(Y_final - Y_true, dim=1).square()
        
        return {
            # Objective metrics
            'objective': obj_pred.mean().item(),
            'true_objective': obj_true.mean().item(),
            'opt_gap_mean': opt_gap.mean().item(),
            'opt_gap_std': opt_gap.std().item(),
            'opt_gap_max': opt_gap.max().item(),
            'opt_gap_min': opt_gap.min().item(),
            
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
            'solution_distance_std': solution_distance.std().item(),
            'solution_distance_max': solution_distance.max().item(),
        }
    
    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across batches."""
        if not all_metrics:
            return {}
        
        keys = all_metrics[0].keys() - {'inference_time'}
        aggregated = {}
        
        for key in keys:
            values = [m[key] for m in all_metrics]
            if key.endswith('_std'):
                # For std metrics, compute overall std
                aggregated[key] = np.std([m[key.replace('_std', '_mean')] for m in all_metrics])
            else:
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    def _print_evaluation_summary(self, split_name, metrics):
        """Print evaluation summary."""
        print(f"\n{split_name.upper()} EVALUATION RESULTS:")
        print("=" * 50)
        print(f"Objective Value:     {metrics.get('objective', 0):.6e}")
        print(f"True Objective:      {metrics.get('true_objective', 0):.6e}")
        print(f"Optimality Gap:      {metrics.get('opt_gap_mean', 0):.6e} ± {metrics.get('opt_gap_std', 0):.6e}")
        print(f"Eq Violation l1:   {metrics.get('eq_violation_l1_mean', 0):.6e} (max: {metrics.get('eq_violation_l1_max', 0):.6e})")
        print(f"Ineq Violation l1: {metrics.get('ineq_violation_l1_mean', 0):.6e} (max: {metrics.get('ineq_violation_l1_max', 0):.6e})")
        print(f"Solution Distance:   {metrics.get('solution_distance_mean', 0):.6e} ± {metrics.get('solution_distance_std', 0):.6e}")
        print(f"Avg Inference Time:  {metrics.get('avg_inference_time', 0):.4f}s")
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
        print("=" * 80)
        print(f"{'Batch Size':<12} {'Objective':<12} {'Opt Gap':<12} {'Eq Viol':<12} {'Ineq Viol':<12} {'Time (s)':<10}")
        print("-" * 80)
        
        for batch_size, result in results.items():
            if 'error' in result:
                print(f"{batch_size:<12} {'OOM':<12} {'OOM':<12} {'OOM':<12} {'OOM':<12} {'OOM':<10}")
            else:
                metrics = result['metrics']
                print(f"{batch_size:<12} "
                      f"{metrics.get('objective', 0):<12.4e} "
                      f"{metrics.get('opt_gap_mean', 0):<12.4e} "
                      f"{metrics.get('eq_violation_l1_mean', 0):<12.4e} "
                      f"{metrics.get('ineq_violation_l1_mean', 0):<12.4e} "
                      f"{metrics.get('total_time', 0):<10.2f}")
        
        print("=" * 80)
        




