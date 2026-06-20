# Homework 3 — Model-Based RL

## Setup and Installation

    conda env create -f environment.yml
    conda activate cse579a1


## Running the assignment

    python main.py --model_type single --plan_mode random_mpc
    python main.py --model_type single --plan_mode mppi
    python main.py --model_type ensemble --plan_mode mppi

Append `--test` to evaluate saved checkpoints instead of training.

## Files you need to touch:
More details in the [assignment spec](https://courses.cs.washington.edu/courses/cse579/26sp/projects/homework3/CSE579SP26_HW3.pdf).
- `main.py` (hyperparameter tuning only)
- `planning.py` — random MPC, MPPI, and ensemble MPPI TODOs
- `train_model.py` — ensemble training TODO
