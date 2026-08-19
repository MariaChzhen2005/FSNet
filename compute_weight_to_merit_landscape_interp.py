import os
import yaml
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np

from utils.trainer_2 import load_instance, Trainer

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

PROBLEM_TYPES = ["convex", "nonconvex", "nonsmooth_nonconvex"]
PROBLEM_NAMES = ["qp", "qcqp", "socp"]

eq_weight = 1e5
ineq_weight = 1e5


def create_parser(arg_list=None):
    parser = argparse.ArgumentParser(description="Neural Network Optimization")

    # General parameters
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str)
    parser.add_argument("--prob_type", type=str, choices=PROBLEM_TYPES)
    parser.add_argument("--prob_name", type=str, choices=PROBLEM_NAMES)
    parser.add_argument("--prob_size", type=int, nargs="+", default=[100, 50, 50, 10000])
    parser.add_argument("--network", type=str, default="MLP")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--ablation", type=bool, default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--en_subopt", type=bool, default=False)
    parser.add_argument("--subopt_ratio", type=float, default=0.0)
    parser.add_argument("--save_intermediate", type=bool, default=False)

    # dataset parameters
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_size", type=int)
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--dropout", type=float)

    # Neural network parameters
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_decay", type=float)
    parser.add_argument("--lr_decay_step", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_layers", type=int)

    # Feasibility seeking parameters
    parser.add_argument("--scale", type=float)
    parser.add_argument("--dist_weight", type=float)
    parser.add_argument("--max_diff_iter", type=int)

    args = parser.parse_args(arg_list)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # overrides
    if args.method:
        config["seed"] = args.seed
        config["method"] = args.method
    if args.prob_type:
        config["prob_type"] = args.prob_type
    if args.prob_name:
        config["prob_name"] = args.prob_name
    if args.prob_size:
        config["prob_size"] = args.prob_size
    if args.network:
        config["network"] = args.network

    config["checkpoint"] = args.checkpoint
    config["en_subopt"] = args.en_subopt
    config["subopt_ratio"] = args.subopt_ratio
    config["save_intermediate"] = args.save_intermediate

    # dataset overrides
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.train_size:
        config["train_size"] = args.train_size
    if args.val_size:
        config["val_size"] = args.val_size
    if args.test_size:
        config["test_size"] = args.test_size

    # NN overrides
    if args.lr:
        config["lr"] = args.lr
    if args.lr_decay:
        config["lr_decay"] = args.lr_decay
    if args.lr_decay_step:
        config["lr_decay_step"] = args.lr_decay_step
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    if args.hidden_dim:
        config["hidden_dim"] = args.hidden_dim
    if args.num_layers:
        config["num_layers"] = args.num_layers
    if args.dropout:
        config["dropout"] = args.dropout

    # FS params
    if args.scale:
        config["FSNet"]["scale"] = args.scale
        config["S3Net"]["scale"] = args.scale
        config["semi"]["scale"] = args.scale
    if args.dist_weight is not None:
        config["FSNet"]["dist_weight"] = args.dist_weight
        config["S3Net"]["dist_weight"] = args.dist_weight
        config["semi"]["dist_weight"] = args.dist_weight
    if args.max_diff_iter is not None:
        config["FSNet"]["max_diff_iter"] = args.max_diff_iter
        config["S3Net"]["max_diff_iter"] = args.max_diff_iter
        config["semi"]["max_diff_iter"] = args.max_diff_iter

    config["ablation"] = args.ablation
    return args, config


# -----------------------------------------------------------------------------
# Merit evaluation (scalar over loader)
# -----------------------------------------------------------------------------
def make_eval_merit(base_trainer):
    def eval_merit(net, loader):
        net.eval()
        val_metrics = base_trainer.evaluator.evaluate(net, loader)
        merit = (
            val_metrics["objective"]
            + eq_weight * val_metrics["eq_violation_l1_mean"]
            + ineq_weight * val_metrics["ineq_violation_l1_mean"]
        )
        return float(merit)
    return eval_merit


# -----------------------------------------------------------------------------
# Weights utils
# -----------------------------------------------------------------------------
def clone_weights(net):
    return [p.detach().clone() for p in net.parameters()]

@torch.no_grad()
def set_weights_from_list(net, weights):
    for p, w in zip(net.parameters(), weights):
        p.copy_(w)

@torch.no_grad()
def set_weights_interpolate(net, wA, wB, t):
    # theta(t) = (1-t)*A + t*B
    tt = float(t)
    for p, a, b in zip(net.parameters(), wA, wB):
        p.copy_((1.0 - tt) * a + tt * b)


@torch.no_grad()
def compute_merit_interpolation_1d(
    modelA,
    modelB,
    test_loader,
    eval_merit,
    *,
    tmin=0.0,
    tmax=1.0,
    tnum=51,
    device=None,
    verbose=True,
):
    """
    Evaluate merit along linear interpolation between modelA and modelB.
    Returns:
      t: [tnum] numpy
      merit: [tnum] numpy
    """
    if device is None:
        device = DEVICE

    # Put both models on device
    modelA = modelA.to(device).eval()
    modelB = modelB.to(device).eval()

    # Use a single mutable model to avoid allocating new modules
    # We'll interpolate weights into `modelA` (you can pick either).
    wA = clone_weights(modelA)
    wB = clone_weights(modelB)

    # Save original weights of modelA to restore after
    w0 = clone_weights(modelA)

    ts = np.linspace(tmin, tmax, tnum, dtype=np.float32)
    merits = np.empty((tnum,), dtype=np.float32)

    for k, t in enumerate(ts, start=1):
        set_weights_interpolate(modelA, wA, wB, t)
        m = eval_merit(modelA, test_loader)
        merits[k - 1] = float(m)

        if verbose and (k % max(1, tnum // 20) == 0):
            print(f"[{k:>4}/{tnum}] t={t:.4f} merit={merits[k-1]:.6g}", flush=True)

    # Restore
    set_weights_from_list(modelA, w0)
    return ts, merits


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Base trainer for evaluator + dataset
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--batch_size", "256",
    ])
    opt_problem, result_save_dir = load_instance(config)
    base_trainer = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)

    test_loader = DataLoader(
        base_trainer.opt_problem.test_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    eval_merit = make_eval_merit(base_trainer)

    # Load model_A0 (EDIT PATH)
    
    # base model after initial supervised training
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--checkpoint",
        "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260115-184556_MLP_sup_pen_seed0_nepochs1000_lr0.0001_trainsize7000_subopt_3_3.0/model_150.pt",
    ])

    # base model random
    # args, config = create_parser([
    #     "--method", "FSNet",
    #     "--prob_type", "nonsmooth_nonconvex",
    #     "--prob_name", "socp",
    # ])

    opt_problem, result_save_dir = load_instance(config)
    trainerA = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)
    modelA = trainerA.train().to(DEVICE)

    # Load model_B0 (EDIT PATH)

    # final vanilla model
    # args, config = create_parser([
    #     "--method", "FSNet",
    #     "--prob_type", "nonsmooth_nonconvex",
    #     "--prob_name", "socp",
    #     "--checkpoint",
    #     "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000/model.pt",
    # ])

    # final warmstarted model
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--checkpoint",
        "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260117-191337_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150/model.pt",
    ])

    opt_problem, result_save_dir = load_instance(config)
    trainerB = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)
    modelB = trainerB.train().to(DEVICE)

    # Compute 1D merit along interpolation
    t, merit = compute_merit_interpolation_1d(
        modelA,
        modelB,
        test_loader,
        eval_merit,
        tmin=-0.5,   # extends "before A"
        tmax=1.5,    # extends "past B"
        tnum=201,    # higher resolution since range is wider
        device=DEVICE,
        verbose=True,
    )

    # Save
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/merit_interpolation_warmstarted.npz"
    np.savez(out_path, t=t, merit=merit)
    print(f"Saved interpolation data to: {out_path}", flush=True)