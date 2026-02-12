# PyTNP

Julia package that wraps a Transformer Neural Process (TNP) implemented in Python. The Julia API uses PythonCall.jl for training, inference, and basic evaluation workflows.

## Highlights
- Python TNP model with Julia interface for loading, training, and prediction
- GP sampler and evaluation utilities in Python, callable from Julia
- Example Julia script that trains (or loads) and plots results with CairoMakie

## Repository Layout
- src/ : Julia module and Python implementation
- scripts/ : training and example scripts
- data/ : model weights and generated outputs

## Setup
1. Create a Python virtual environment and install Python deps (torch, numpy, matplotlib).
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib
```
2. Add the Julia package (installs PythonCall.jl and CairoMakie.jl).
```julia
using Pkg
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/venv/bin/python"
Pkg.add("PyTNP")
```

## Julia Usage
See the example usage in "scripts/example.jl".
```bash
julia scripts/example.jl
```

Minimal API usage in Julia:
```julia
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/venv/bin/python"
using PyTNP

PyTNP.train_model(num_iterations=1000, save_path="data/tnp_model.pt")
model = PyTNP.load_model("data/tnp_model.pt")

context_x = [-1.0, 0.0, 1.0]
context_y = [0.5, 0.0, -0.2]
target_x = collect(range(-2.0, 2.0, length=100))

mean, std = PyTNP.predict(model, context_x, context_y, target_x)
```

## Python Usage
Train and evaluate with the Python script:
```bash
python scripts/train_and_eval.py
```
