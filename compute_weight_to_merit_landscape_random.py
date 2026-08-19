import os
import yaml
import torch
from torch.utils.data import DataLoader
import argparse
from utils.trainer_2 import load_instance, Trainer
import copy
import numpy as np
import matplotlib.pyplot as plt

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
# Merit evaluation (assumes evaluator already handles device placement)
# -----------------------------------------------------------------------------
# def make_eval_merit(base_trainer):
#     def eval_merit(net, loader):
#         net.eval()
#         val_metrics = base_trainer.evaluator.evaluate(net, loader)
#         merit = (
#             val_metrics["objective"]
#             + eq_weight * val_metrics["eq_violation_l1_mean"]
#             + ineq_weight * val_metrics["ineq_violation_l1_mean"]
#         )
#         return merit
#     return eval_merit

def make_eval_merit(base_trainer):
    return base_trainer.evaluator.evaluate_merit

def make_loss_merit(base_trainer):
    return base_trainer.evaluator.evaluate_loss

# -----------------------------------------------------------------------------
# Device-consistent directions + weight perturbation
# -----------------------------------------------------------------------------
def clone_weights(net):
    return [p.detach().clone() for p in net.parameters()]

def get_random_weights_like_params(params, seed=1):
    """
    Generate reproducible random tensors with the same shapes/dtypes/devices
    as `params`.
    """
    # Use a local RNG so global state is untouched
    device = params[0].device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    return [
        torch.randn(
            p.shape,
            device=p.device,
            dtype=p.dtype,
            generator=g,
        )
        for p in params
    ]

def normalize_directions_for_weights(direction, weights, norm="filter", ignore="biasbn"):
    assert len(direction) == len(weights)
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == "biasbn":
                d.zero_()
            else:
                d.copy_(w)
        else:
            if norm == "filter":
                d.mul_(w.norm() / (d.norm() + 1e-10))
            elif norm == "layer":
                d.mul_(w.norm() / (d.norm() + 1e-10))
            elif norm == "weight":
                d.mul_(w)
            elif norm == "dfilter":
                d.div_(d.norm() + 1e-10)
            elif norm == "dlayer":
                d.div_(d.norm() + 1e-10)
            else:
                raise ValueError(f"Unknown norm={norm}")

def create_random_direction(net, ignore="biasbn", norm="filter", seed=1):
    params = list(net.parameters())
    direction = get_random_weights_like_params(params, seed=seed)
    normalize_directions_for_weights(direction, params, norm=norm, ignore=ignore)
    return direction

@torch.no_grad()
def set_weights(net, base_weights, directions=None, step=None):
    if directions is None:
        for p, w0 in zip(net.parameters(), base_weights):
            p.copy_(w0)
        return

    assert step is not None, "step must be specified if directions is specified"

    if len(directions) == 2:
        dx, dy = directions
        ax, ay = float(step[0]), float(step[1])
        for p, w0, d0, d1 in zip(net.parameters(), base_weights, dx, dy):
            p.copy_(w0 + ax * d0 + ay * d1)
    else:
        (dx,) = directions
        a = float(step)
        for p, w0, d in zip(net.parameters(), base_weights, dx):
            p.copy_(w0 + a * d)

def _to_scalar(x):
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    if isinstance(x, dict):
        if "merit" in x:
            return _to_scalar(x["merit"])
        for v in x.values():
            try:
                return _to_scalar(v)
            except Exception:
                pass
        raise ValueError(f"Cannot convert eval output dict to scalar: {list(x.keys())}")
    raise TypeError(f"Unsupported eval output type: {type(x)}")

@torch.no_grad()
def compute_merit_surface_2d(
    model,
    test_loader,
    eval_merit,
    *,
    xmin=-1.0, xmax=1.0, xnum=51,
    ymin=-1.0, ymax=1.0, ynum=51,
    ignore="biasbn",
    norm="filter",
    device=None,
    verbose=True,
):
    model.eval()
    if device is not None:
        model.to(device)

    w0 = clone_weights(model)
    dx = create_random_direction(model, ignore=ignore, norm=norm, seed=5)
    dy = create_random_direction(model, ignore=ignore, norm=norm, seed=6)

    xs = np.linspace(xmin, xmax, xnum, dtype=np.float32)
    ys = np.linspace(ymin, ymax, ynum, dtype=np.float32)
    Z = np.empty((ynum, xnum), dtype=np.float32)

    total = xnum * ynum
    k = 0

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            set_weights(model, w0, directions=[dx, dy], step=(x, y))
            m = eval_merit(model, test_loader)
            Z[j, i] = _to_scalar(m)

            k += 1
            if verbose and (k % max(1, total // 20) == 0):
                print(f"[{k:>6}/{total}] x={x:+.3f}, y={y:+.3f}, merit={Z[j,i]:.6g}")

    set_weights(model, w0)
    X, Y = np.meshgrid(xs, ys)
    return X, Y, Z, dx, dy


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Build base_trainer (for evaluator + datasets)
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
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

    eval_value = make_eval_merit(base_trainer)
    # eval_value = make_loss_merit(base_trainer)

    # # Load model_A0
    # args, config = create_parser([
    #     "--method", "FSNet",
    #     "--prob_type", "nonsmooth_nonconvex",
    #     "--prob_name", "socp",
    #     "--checkpoint",
    #     "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260117-191337_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000_finetune_20260115-184556_sup_seedpen_model_150/model.pt",
    # ])
    # opt_problem, result_save_dir = load_instance(config)
    # trainer_dummy = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)
    # model_A0 = trainer_dummy.train()
    # model_A0 = model_A0.to(DEVICE)

    # Load model_B0
    args, config = create_parser([
        "--method", "FSNet",
        "--prob_type", "nonsmooth_nonconvex",
        "--prob_name", "socp",
        "--checkpoint",
        "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000/20260115-013254_MLP_FSNet_seed3_nepochs300_lr0.0001_trainsize7000/model.pt",
    ])
    opt_problem, result_save_dir = load_instance(config)
    trainer_dummy = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)
    model_B0 = trainer_dummy.train()
    model_B0 = model_B0.to(DEVICE)

    # Compute + save landscape around model_B0
    X, Y, Z, dx, dy = compute_merit_surface_2d(
        model_B0,
        test_loader,
        eval_value,
        xmin=-1, xmax=1, xnum=51,
        ymin=-1, ymax=1, ynum=51,
        ignore="biasbn",
        norm="filter",
        device=DEVICE,
        verbose=True,
    )

    os.makedirs("figures", exist_ok=True)
    npz_dir = f"figures/model_vanilla_merit_surface_data2101.npz"
    np.savez(npz_dir, X=X, Y=Y, Z=Z)
    print(f"Saved surface data to: {npz_dir}")