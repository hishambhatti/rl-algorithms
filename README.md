# rl-algorithms

PyTorch implementations of reinforcement learning and imitation learning algorithms, trained and evaluated in MuJoCo environments via [Gymnasium](https://gymnasium.farama.org/).

## Overview

This repo collects three self-contained modules that build on each other — from learning from demonstrations, to model-free control, to planning with learned dynamics:

| Module | Topic | Algorithms | Environments |
|--------|-------|------------|--------------|
| [`imitation-learning/`](imitation-learning/) | Imitation learning | Behavior cloning, DAgger, autoregressive policies, diffusion policies | [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/), [PointMaze](https://robotics.farama.org/envs/maze/point_maze/) |
| [`model-free-rl/`](model-free-rl/) | Model-free RL | Policy gradient, actor-critic, soft actor-critic (SAC) | Inverted Pendulum, Ant |
| [`model-based-rl/`](model-based-rl/) | Model-based RL | Random MPC (shooting), MPPI, ensemble MPPI | Reacher |

Each module includes runnable training scripts, Jupyter notebooks with experiments and plots, and a short README with setup instructions.

## Highlights

**Imitation learning** — Train Gaussian, autoregressive, and diffusion policies to mimic expert trajectories. Compare behavior cloning against DAgger on distribution-shift-heavy tasks like maze navigation.

**Policy optimization** — Implement REINFORCE with a learned baseline, a single-Q actor-critic agent, and SAC with double Q-networks and automatic entropy tuning.

**Model-based planning** — Learn a dynamics model from rollouts, then plan with random shooting, MPPI, and an ensemble variant that averages predictions across multiple models for more robust control.

## Quick start

Each module has its own conda environment file. From any module directory:

```bash
conda env create -f environment.yml
conda activate cse579a1
```

**Imitation learning**
```bash
cd imitation-learning
python main.py --env reacher --train behavior_cloning --policy gaussian
python main.py --env pointmaze --train dagger --policy diffusion
```

**Model-free RL**
```bash
cd model-free-rl
python main.py --task pg --env pendulum
python main.py --task sac --env ant
```

**Model-based RL**
```bash
cd model-based-rl
python main.py --model_type single --plan_mode mppi
python main.py --model_type ensemble --plan_mode mppi
```

See each module's README for the full set of flags and options.

## Stack

- Python 3.10, PyTorch 2.6
- Gymnasium + MuJoCo
- Conda for environment management

---

*Completed as part of CSE 579 (Reinforcement Learning) at the University of Washington, Spring 2026.*
