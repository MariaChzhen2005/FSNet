## FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees
This repository is by 
[Hoang T. Nguyen](https://www.linkedin.com/in/hoang-nguyen-971519201/) and 
[Priya L. Donti](https://www.priyadonti.com)
 and contains source code to reproduce the experiments in our paper 
 ["FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees"](https://arxiv.org/abs/2506.00362).


## Abstract
<p style="text-align: justify;">
Efficiently solving constrained optimization problems is crucial for numerous real-world applications, yet traditional solvers are often computationally prohibitive for real-time use. Machine learning-based approaches have emerged as a promising alternative to provide approximate solutions at faster speeds, but they struggle to strictly enforce constraints, leading to infeasible solutions in practice. To address this, we propose the Feasibility-Seeking-Integrated Neural Network (FSNet), which integrates a feasibility-seeking step directly into its solution procedure to ensure constraint satisfaction. This feasibility-seeking step solves an unconstrained optimization problem that minimizes constraint violations in a differentiable manner, enabling end-to-end training and providing guarantees on feasibility and convergence. Our experiments across a range of different optimization problems, including both smooth/nonsmooth and convex/nonconvex problems, demonstrate that FSNet can provide feasible solutions with solution quality comparable to (or in some cases better than) traditional solvers, at significantly faster speeds. 

<p align="center">
  <img src="figures\diagram.png" alt="FSNet Diagram" width="800"/>
</p>


If you find this repository helpful in your publications, please consider citing our paper.
```bash
@article{nguyen2025fsnet,
    title={FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees}, 
    author={Hoang T. Nguyen and Priya L. Donti},
    year={2025},
    journal={arXiv preprint arXiv:2506.00362},
}
```


## 🚀 Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎓 Usage

### Training and Test

```bash
python main.py \
  --method <FSNet|penalty|adaptive_penalty|DC3|projection> \
  --prob_type <convex|nonconvex|nonsmooth_nonconvex> \
  --prob_name <qp|qcqp|socp>
```

* `--method`

  * `FSNet`              (Feasibility-Seeking Neural Network)
  * `penalty`            (Penalty method)
  * `adaptive_penalty`   (Adaptive Penalty method)
  * `DC3`                (Deep Constraint Completion and Correction)
  * `projection`         (Projection-based method; supported for QP only)
* `--prob_type`

  * `convex`
  * `nonconvex`
  * `nonsmooth_nonconvex`
* `--prob_name`

  * `qp`   (Quadratic Program)
  * `qcqp` (Quadratically Constrained Quadratic Program)
  * `socp` (Second-Order Cone Program)
* And see `main.py` for more relevant flags.

Example:
```bash
python main.py --method FSNet --prob_type convex --prob_name qp
python main.py --method FSNet --prob_type nonconvex --prob_name qp
python main.py --method FSNet --prob_type nonconvex --prob_name socp --dropout 0.1

```

Training and evaluation use `tqdm` progress bars for loss, objective value,
relative optimality gap, feasibility rate, constraint violations, and elapsed
time. History records include both raw and relative objective gaps. The
feasibility rate uses `feasibility_tol` from the config (default `1e-5`) and can
be overridden with `--feasibility_tol`.

Each run writes metrics under its result directory in two portable formats:

* `metrics_seed<seed>.jsonl` is updated after every train, validation, and test
  phase, so partial results remain available if a run is interrupted.
* `metrics_seed<seed>.json` contains the consolidated history after the run.

The existing `results_seed<seed>.pkl` also contains the complete metric history
and detailed test predictions.

### Architecture merit-landscape ablation

Train and compare the four-layer, width-64 MLP, ICNN, and residual-MLP FSNet
backbones on the nonsmooth nonconvex SOCP dataset with one command:

```bash
python compute_weight_to_merit_landscape_random.py all
```

The default landscape configuration uses training seed `2025`, matched random
direction seeds `0 1 2 3 4`, a detailed `31 x 31` grid of actual merit
evaluations per surface, 300-DPI rendering, filter-normalized directions over
`[-1, 1]^2`, and the merit function

```text
objective + 1e4 * equality_L1_violation + 1e4 * inequality_L1_violation
```

Checkpoints are written below `results/landscape_ablation/`. Raw surfaces,
individual 3D plots, matched-seed 3D comparison plots, an all-model 3D
comparison, and a JSON manifest are written to `figures/`. Existing compatible
checkpoints and surface data are reused automatically. The camera can be
adjusted with `--elevation` and `--azimuth`. Every plot uses raw merit on the
z-axis and one shared raw-merit color scale; no shifting or log transform is
applied.

The default full comparison computes 15 surfaces and 14,415 parameter points.
That is intentionally detailed and can take substantial time because every
point runs FSNet's feasibility refinement. Completed surfaces are cached. Use
`--grid-size 41` for a very high-resolution run, or the smoke test below before
committing to the full computation.

The stages can also be run separately:

```bash
python compute_weight_to_merit_landscape_random.py train
python compute_weight_to_merit_landscape_random.py visualize
```

For a quick pipeline smoke test before the full run:

```bash
python compute_weight_to_merit_landscape_random.py all \
  --num-epochs 2 \
  --grid-size 5 \
  --landscape-test-size 16 \
  --checkpoint-root results/landscape_ablation_smoke \
  --output-dir figures/landscape_smoke
```

Use `--force-train` or `--force-landscape` to replace compatible cached output,
and run the script with `--help` for grid, seed, checkpoint, and output options.
