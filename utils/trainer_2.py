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
import ipdb

from utils.optimization_utils import *
from utils.lbfgs import nondiff_lbfgs_solve, hybrid_lbfgs_solve
from models.neural_networks import MLP
from utils.evaluator import Evaluator

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)


def load_instance(config):
    """Loads problem instance, data, and sets up save directory."""

    # Load data
    seed = config['seed']
    method = config['method']
    train_size = config['train_size']
    val_size = config['val_size']
    test_size = config['test_size']
    en_subopt = config['en_subopt']
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
    dataset_filepath = os.path.join(
        'datasets', 
        prob_type, 
        prob_name,
        f"random{seed_data}_{prob_name}_dataset_var{prob_size[0]}_ineq{prob_size[1]}_eq{prob_size[2]}_ex{prob_size[3]}"
    )

    if method == "sup" or method == "sup_partial" or method == "sup_pen":
        if config['en_subopt'] == 1:
            dataset_filepath = dataset_filepath + f'_subopt_noise{config["subopt_ratio"]}_bias{config["subopt_ratio"]}'
        if config['en_subopt'] == 2:
            if config['subopt_ratio'] == 1:
                dataset_filepath = dataset_filepath + f'_tol1e0_ready'
            if config['subopt_ratio'] == -10:
                dataset_filepath = dataset_filepath + f'_tol1em1_ready'

    print("Loading  dataset from:", dataset_filepath, '\n')
    with open(dataset_filepath, 'rb') as f:
        dataset = pickle.load(f)

    # Create problem instance using the appropriate class
    opt_problem = problem_names[prob_name](dataset, train_size, val_size, test_size, seed, en_subopt) 

    opt_problem.device = DEVICE
    print("Running on: ", DEVICE)
    for attr in dir(opt_problem):
        var = getattr(opt_problem, attr)
        if torch.is_tensor(var):
            try:
                setattr(opt_problem, attr, var.to(DEVICE))
            except AttributeError:
                pass

    if config['ablation'] == True:
        result_save_dir = os.path.join('ablation_results', prob_type, prob_name, str(opt_problem), config['network'] + '_' + config['method'], 'dist_'+ str(config['FSNet']['dist_weight']) + '_diff_' + str(config['FSNet']['max_diff_iter']))
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_save_dir = os.path.join('results', prob_type, prob_name, str(opt_problem), timestamp + '_' + config['network'] + '_' + config['method'] + '_seed' + str(seed) + '_dropout' + str(config['dropout']))

        if config['en_subopt'] != 0:
            result_save_dir += f"_subopt_{config['en_subopt']}_{config['subopt_ratio']}"

        if config['checkpoint']:
            # assmume checkpoint path format contains date and other info results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20251004-214029_MLP_sup_seed0_dropout0.1/model_580.pt
            ckpt_date = config['checkpoint'].split('/')[4].split('_')[0]
            ckpt_number = config['checkpoint'].split('_')[-1].split('.')[0]
            result_save_dir += f"_finetune_{ckpt_date}_model_{ckpt_number}"

    # if not os.path.exists(result_save_dir):
    #     os.makedirs(result_save_dir)
    
    return opt_problem, result_save_dir


def create_model(opt_problem, method, config):
    """Creates and returns a neural network model."""
    
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    network = config['network']
    dropout = config["dropout"]

    if network == 'MLP':
        if method == "DC3" or method == "sup_partial":
            out_dim = opt_problem.partial_vars.shape[0]
            model = MLP(opt_problem.xdim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout)
        else:
            model = MLP(opt_problem.xdim, hidden_dim, opt_problem.ydim, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model}")
    return model.to(DEVICE)


class Trainer:
    def __init__(self, opt_problem, config, save_dir=None):
        """Initializes the Trainer with opt_problem, method, and configuration."""
        self.opt_problem = opt_problem
        self.method = config['method']
        self.config = config
        self.save_dir = save_dir
        
        self.config_method = config[self.method]
        self.evaluator = Evaluator(opt_problem, self.method, config)
        self.en_feasibility = False
        self.en_penalty = False
        
        self._initialize_params()

    def compute_batch_loss(self, X_batch: torch.Tensor, Y_pred: torch.Tensor, Y_label: torch.Tensor, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the loss and additional metrics."""
        Y_pred_scaled = self.opt_problem.scale(Y_pred)
        metrics = {}
        if self.method == "penalty":
            return self._penalty_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "adaptive_penalty":
            return self._adaptive_penalty_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "FSNet":
            return self._fsnet_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "S3Net":            
            return self._s3net_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "semi":            
            return self._semi_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "sup":
            return self._sup_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "sup_partial":
            return self._sup_partial_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "sup_pen":
            return self._sup_pen_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "DC3": 
            return self._dc3_loss(X_batch, Y_pred_scaled, metrics)            
        elif self.method == "projection":
            return self._projection_loss(X_batch, Y_pred_scaled, metrics)
        

    def _penalty_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the penalty loss."""
        obj = self.opt_problem.obj_fn(Y_pred_scaled)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
    
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
        obj = self.opt_problem.obj_fn(Y_pred_scaled)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)

        loss = self.config_method['obj_weight'] * obj + \
               self.adaptive_eq_weight * eq_violation + \
               self.adaptive_ineq_weight * ineq_violation

        with torch.no_grad():
            self.adaptive_eq_weight = torch.clamp(self.adaptive_eq_weight + self.config_method['increasing_rate'] * eq_violation.mean(), min=0.0, max=self.config_method['eq_pen_weight_max'])
            self.adaptive_ineq_weight = torch.clamp(self.adaptive_ineq_weight + self.config_method['increasing_rate'] * ineq_violation.mean(), min=0.0, max=self.config_method['ineq_pen_weight_max'])
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
    
    def _penalized_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the penalty loss."""
        obj = self.opt_problem.obj_fn(Y_pred_scaled)

        ineq_violation = self.opt_problem.ineq_resid(
            X_batch, Y_pred_scaled).square().sum(dim=1)

        loss = obj + 1e-3 * ineq_violation

        return loss

    def _fsnet_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the FSNet loss."""
        pre_eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        pre_ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        # Feasibility refinement using hybrid L-BFGS
        Y_final = hybrid_lbfgs_solve(
            X_batch,
            Y_pred_scaled,
            self.opt_problem,
            val_tol=self.config_method['val_tol'],
            memory=self.config_method['memory_size'],
            max_iter=self.config_method['max_iter'],
            max_diff_iter=self.config_method['max_diff_iter'],
            scale=self.config_method['scale'],
        )
        obj = self.opt_problem.obj_fn(Y_final)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

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
            'distance': distance.mean().item(),
        })
        return loss, metrics

    def _sup_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, Y_label: torch.Tensor, metrics: Dict, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:

        Y_final = Y_pred_scaled
            
        obj = self.opt_problem.obj_fn(Y_final)
        
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()

        # per-sample robust supervised loss
        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)

        loss = huber(Y_final - Y_label).mean(dim=1)  # [B]

        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics

    def _sup_partial_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, Y_label: torch.Tensor, metrics: Dict, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        Y_final = Y_pred_scaled

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()
        obj = distance
        eq_violation = distance
        ineq_violation = distance
        eq_violation_l1 = distance
        ineq_violation_l1 = distance
        
        # per-sample robust supervised loss
        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)

        loss = huber(Y_final - Y_label[:, self.opt_problem.partial_vars]).mean(dim=1)  # [B]

        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics
    
    def _sup_pen_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, Y_label: torch.Tensor, metrics: Dict, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        pre_eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        pre_ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        Y_final = Y_pred_scaled
            
        obj = self.opt_problem.obj_fn(Y_final)
        
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()

        sup_weight = 1.0

        # per-sample robust supervised loss
        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)

        loss = sup_weight * huber(Y_final - Y_label).mean(dim=1)  # [B]

        loss_eq_term = self.config_method['eq_pen_weight'] * pre_eq_violation 
        loss_ineq_term = self.config_method['ineq_pen_weight'] * pre_ineq_violation

        self.en_penalty = True
        if self.en_penalty:           
            loss += loss_eq_term + loss_ineq_term
            
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics
    
    def _s3net_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, Y_label: torch.Tensor, metrics: Dict, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        pre_eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        pre_ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        if epoch_metrics['epoch'] > -1:#0.1 * self.config['num_epochs']:
            if self.en_penalty == False:
                print("Enabling penalty terms")
            self.en_penalty = True

        if epoch_metrics['epoch'] > 3:#0.2 * self.config['num_epochs']:
            if self.en_feasibility == False:
                print("Enabling feasibility seeking")
            self.en_feasibility = True
            # self.en_penalty = True

        if self.en_feasibility:
            # Feasibility refinement using hybrid L-BFGS
            # self.config_method['max_iter'] = 50
            Y_final = hybrid_lbfgs_solve(
                X_batch,
                Y_pred_scaled,
                self.opt_problem,
                val_tol=self.config_method['val_tol'],
                memory=self.config_method['memory_size'],
                max_iter=self.config_method['max_iter'],
                max_diff_iter=self.config_method['max_diff_iter'],
                scale=self.config_method['scale'],
            )
        else:
            Y_final = Y_pred_scaled
            
        obj = self.opt_problem.obj_fn(Y_final)
        
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()

        sup_weight = 1.0

        if self.en_penalty:
            sup_weight *= 0.5

        # per-sample robust supervised loss
        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)
        # loss = sup_weight * ((Y_final - Y_label) ** 2).sum(dim=1, keepdim=True).squeeze()  # [B, 1]
        loss = sup_weight * huber(Y_final - Y_label).mean(dim=1)  # [B]
        # loss = sup_weight * ((Y_final - Y_label).abs()).mean(dim=1)  # [B]

        loss_obj_term = self.config_method['obj_weight'] * obj 
        loss_dist_term = self.config_method['dist_weight'] * distance 
        loss_eq_term = self.config_method['eq_pen_weight'] * pre_eq_violation 
        loss_ineq_term = self.config_method['ineq_pen_weight'] * pre_ineq_violation
        
        # import ipdb; ipdb.set_trace()

        if self.en_penalty:
            if pre_eq_violation.mean() >= 1e3 or pre_ineq_violation.mean() >= 1e3:              
                loss += loss_obj_term + loss_dist_term + loss_eq_term + loss_ineq_term
            else:
                loss += loss_obj_term + loss_dist_term
            
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics
    
    def _semi_loss(self, X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics):
        if epoch_metrics['epoch'] > -1:#0.1 * self.config['num_epochs']:
            if self.en_penalty == False:
                print("Enabling penalty terms")
            self.en_penalty = True

        if epoch_metrics['epoch'] > -1:#0.2 * self.config['num_epochs']:
            if self.en_feasibility == False:
                print("Enabling feasibility seeking")
            self.en_feasibility = True

        B = X_batch.shape[0]
        idx_sup = torch.randperm(B, device=X_batch.device)[:B // 2]
        idx_unsup = torch.tensor([i for i in range(B) if i not in idx_sup], device=X_batch.device)

        # --- forward feasibility refinement (DC3 / FSNet-like) ---
        if self.en_feasibility:
            Y_final = hybrid_lbfgs_solve(
                X_batch, Y_pred_scaled, self.opt_problem,
                val_tol=self.config_method['val_tol'],
                memory=self.config_method['memory_size'],
                max_iter=self.config_method['max_iter'],
                max_diff_iter=self.config_method['max_diff_iter'],
                scale=self.config_method['scale'],
            )
        else:
            Y_final = Y_pred_scaled

        # --- compute objectives and constraints ---
        obj = self.opt_problem.obj_fn(Y_final)
        eq_viol = self.opt_problem.eq_resid(X_batch, Y_final)
        ineq_viol = self.opt_problem.ineq_resid(X_batch, Y_final)
        eq_violation = eq_viol.square().sum(dim=1)
        ineq_violation = ineq_viol.square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)
        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()

        # --- supervised subset ---
        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)

        # sup_weight = np.exp(-epoch_metrics['epoch'] / (0.3 * self.config['num_epochs']))
        sup_weight = 1.0
        loss_sup = torch.zeros(B, device=X_batch.device)
        loss_sup[idx_sup] = sup_weight * huber(Y_final[idx_sup] - Y_label[idx_sup]).mean(dim=1)

        # --- self-supervised subset ---
        loss_unsup = (
            self.config_method['obj_weight'] * obj +
            self.config_method['dist_weight'] * distance +
            self.config_method['eq_pen_weight'] * eq_violation +
            self.config_method['ineq_pen_weight'] * ineq_violation
        )
        loss_unsup[idx_sup] = 0.0  # only apply unsupervised terms to unlabeled subset

        # --- total loss ---
        loss = loss_sup + loss_unsup
        loss = loss.mean()

        # --- metric logging ---
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics
            
    def _dc3_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the DC3 loss."""
        Y_completion = self.opt_problem.complete_partial(X_batch, Y_pred_scaled)
        Y_final = grad_steps(self.opt_problem, X_batch, Y_completion, self.config)
        obj = self.opt_problem.obj_fn(Y_final)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)
        
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
        Y_final = self.opt_problem.qpth_projection(X_batch, Y_pred_scaled)
        obj = self.opt_problem.obj_fn(Y_final)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)
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
        epoch_metrics = {'obj': 0.0, 'loss': 0.0, 'eq_violation': 0.0, 'ineq_violation': 0.0, 'eq_violation_l1': 0.0, 'ineq_violation_l1': 0.0, 'distance': 0.0, 'epoch': epoch}
        
        # Update method parameters if needed
        # self._update_epoch_params(epoch)
        
        for batch_idx, (X_batch, Y_label) in enumerate(train_loader):
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            Y_label = Y_label.to(DEVICE, non_blocking=True)
            Y_pred = self.model(X_batch)
            
            loss, batch_metrics = self.compute_batch_loss(X_batch, Y_pred, Y_label, epoch_metrics)
            
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
        epoch_metrics['epoch'] *= num_batches
            
        return epoch_metrics
    
    def _initialize_params(self) -> None:
        if self.method == 'adaptive_penalty' or self.method == 'sup_pen':
            self.adaptive_eq_weight = self.config_method['eq_pen_weight']
            self.adaptive_ineq_weight = self.config_method['ineq_pen_weight']
           
    def _update_epoch_params(self, epoch: int) -> None:
        """Update parameters based on epoch."""
        # FSNet tolerance decay
        if ((self.method == 'FSNet' or self.method == 'S3Net' or self.method == 'semi') and (epoch + 1) % self.config_method['decay_tol_step'] == 0):
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
    
 
    def train(self):
        """Main training loop with detailed results collection."""

        # Prepare data loaders
        batch_size = self.config['batch_size']
        train_size = len(self.opt_problem.train_dataset)
        if train_size <= 50:
            batch_size = 16
        elif train_size <= 100:
            batch_size = 32
        elif train_size <= 500:
            batch_size = 64
        elif train_size <= 1000:
            batch_size = 128
        elif train_size <= 5000:
            batch_size = 256
        self.config['batch_size'] = batch_size
        print(f"Using batch size: {self.config['batch_size']}")

        train_loader = DataLoader(
            self.opt_problem.train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
        )

        val_loader = DataLoader(
            self.opt_problem.val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Initialize model
        if self.config['checkpoint']:
            print(f"Loading model from checkpoint: {self.config['checkpoint']}")
            model_save_content = torch.load(
                self.config['checkpoint'],
                map_location=DEVICE,
                weights_only=False
            )
            model_save_content['config']['dropout'] = self.config['dropout']  # Ensure dropout is set correctly
            self.model = create_model(self.opt_problem, self.method, model_save_content['config'])
            self.model.load_state_dict(model_save_content['model_state_dict'])
        else:
            self.model = create_model(self.opt_problem, self.method, self.config)
        
        # Initialize optimizer and scheduler (fix the initialization issue)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config_method['lr'], 
            weight_decay=0.001, 
            fused=True
        )

        if self.config['checkpoint']:
            warmup_steps = len(train_loader)
            total_steps = len(train_loader) * self.config_method['num_epochs']
            s1 = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_steps)
            s2 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-3)
            self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[s1, s2], milestones=[warmup_steps])
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.config_method['lr_decay_step'], 
                gamma=self.config_method['lr_decay']
            )

        print(f"\nlr: {self.config_method['lr']}, weight_decay: {0.001}, num_epochs: {self.config_method['num_epochs']}\n")

        # # Training history
        # train_history = []
        # val_history = []

        # train_start = time.time()
        # for epoch in range(self.config_method['num_epochs']):
        #     self._update_epoch_params(epoch)
        #     epoch_start = time.time()
            
        #     # Train for one epoch
        #     self.model.train()
        #     epoch_metrics = self.train_epoch(train_loader, epoch)
        #     train_history.append({'epoch': epoch, **epoch_metrics})
        #     epoch_end = time.time()
       
        #     # Log metrics
        #     print(f"Epoch {epoch + 1}/{self.config_method['num_epochs']}, "
        #           f"Loss: {epoch_metrics['loss']:.4f}, "
        #           f"Obj: {epoch_metrics.get('obj', 0):.4f}, "
        #           f"Eq Viol (l1): {epoch_metrics.get('eq_violation_l1', 0):.6f}, "
        #           f"Ineq Viol (l1): {epoch_metrics.get('ineq_violation_l1', 0):.6f}, "
        #           f"Epoch time: {epoch_end - epoch_start:.2f}s")

        #     # Evaluate on validation set
        #     if (epoch) % self.config['eval_step'] == 0:
        #         print(f"\nRunning validation at epoch {epoch}...")
        #         val_metrics = self.evaluator.evaluate(self.model, val_loader, f"validation_epoch_{epoch}")
        #         val_history.append({**val_metrics, 'epoch': epoch})

        #         # Save all results with detailed information
        #         if self.save_dir and self.config['save_intermediate']:
        #             self._save_model(epoch)
        
        # train_end = time.time()
        # training_time = train_end - train_start
        # print(f"\nTraining completed in {training_time:.2f} seconds.")

        # # Enhanced test evaluation with multiple batch sizes and detailed results
        # if hasattr(self.opt_problem, 'test_dataset'):
        #     print("\n" + "="*60)
        #     print("COMPREHENSIVE TEST EVALUATION WITH DETAILED RESULTS")
        #     print("="*60)
            
        #     # Get test batch sizes from config or use defaults
        #     test_batch_sizes = self.config.get('test_batch_sizes', [256, 512])
            
        #     print(f"Testing with batch sizes: {test_batch_sizes}")
            
        #     # Run evaluation with all batch sizes and collect detailed results for all
        #     batch_size_results, all_detailed_results = self.evaluator.evaluate_multiple_batch_sizes(
        #         self.model, 
        #         self.opt_problem.test_dataset, 
        #         test_batch_sizes, 
        #         "test"
        #     )
            
        #     # Combine all test results
        #     final_test_results = {
        #         'batch_size_comparison': batch_size_results,
        #         'detailed_results_all_batch_sizes': all_detailed_results
        #     }
        # else:
        #     print("No test dataset available")
        #     final_test_results = {}
        #     all_detailed_results = None
        
        # # Save all results with detailed information
        # if self.save_dir:
        #     self._save_model_and_results(
        #         train_history, 
        #         val_history, 
        #         final_test_results, 
        #         training_time
        #     )
        
        return self.model
    
    def _save_model(self, epoch: int = None):
        """Saves the model in a .pt file."""
        if not self.save_dir:
            print("Save directory not specified. Skipping saving.")
            return
        
        os.makedirs(self.save_dir, exist_ok=True) # Ensure save directory exists

        # --- 1. Save Model File (.pt) ---
        model_save_content = {
            'model_state_dict': self.model.state_dict(),
            'model_architecture_str': str(self.model), 
            'config': self.config, # Include config for easier model reloading
        }
        model_filename = f"model_{epoch}.pt"
        model_filepath = os.path.join(self.save_dir, model_filename)
        try:
            torch.save(model_save_content, model_filepath)
            print(f"✓ Model saved: {model_filepath}")
        except Exception as e:
            print(f"✗ Error saving model: {e}")

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
        model_filename = f"model.pt"
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

        results_filename = f"results.pkl"
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