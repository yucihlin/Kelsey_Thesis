# post_plot_gpyopt.py
import numpy as np
if not hasattr(np, "bool"):        # NumPy 1.24+ removed np.bool
    np.bool = bool    
import pandas as pd
import matplotlib
matplotlib.use("Agg")
setattr(matplotlib, "numpy", np)      # For GPy to find matplotlib.numpy
import matplotlib.pyplot as plt
import GPy
from GPyOpt.methods import BayesianOptimization as BO
OUT_DIR = "change_seed"         # Directory for CSV files

df = pd.read_csv(f"{OUT_DIR}/all_trials.csv")
# Keep only successful trials
df = df[df["score"].notna()].reset_index(drop=True)

# Construct X, Y for GPyOpt visualization
X_raw = df[["latent","beta"]].to_numpy()
Y_raw = -df["score"].to_numpy().reshape(-1,1)    # ← Usually apply negative sign here


# ==== 1. Reprocess Y: min-max instead of z-score =========
y_min, y_max = Y_raw.min(), Y_raw.max()
Y = (Y_raw - y_min) / (y_max - y_min)      # 0 ~ 1, smaller is worse

# ==== 2. Kernel: Matern52 + Bias + White ===========
k_main = GPy.kern.Matern52(
    input_dim=2, ARD=True,
    variance=2.0,                 # Make μ more sensitive to y amplitude
    lengthscale=[0.2, 0.3]        # Initial values for x1, x2
)
k_main.lengthscale.constrain_bounded(0.2, 3)    # Avoid bamboo curtain or flattening
w = GPy.kern.White(2, variance=2e-2)
w.variance.constrain_bounded(1e-4, 2e-2)   
kernel = (
    k_main
    + GPy.kern.Bias(2, variance=1e-2)         
    # + GPy.kern.White(2, variance=1e-2) 
    +w          
)
k_main.variance.constrain_bounded(0.1, 10) 
domain = [
    {"name": "latent", "type": "continuous", "domain": (10, 35)},
    {"name": "beta",   "type": "continuous", "domain": (0.0, 1.0)}
]

# bo = BO(f=None, domain=domain, X=X, Y=Y, normalize_Y=False)
bo = BO(
    f=None, domain=domain,
    X=X_raw, Y=Y,
    kernel=kernel,
    normalize_Y=False,     # ← y already min-max normalized
    exact_feval=True,
    # acquisition_type='LCB',
    # acquisition_weight=6.0,
    acquisition_type='EI',
    acquisition_par=0.15,
    optimizer_restarts=20
)
# bo.acquisition_type  = 'EI'
# bo.acquisition_par   = 0.15
bo._compute_results()                # Only do GP fitting, no more function runs

# ------- (2) Acquisition + posterior -------------------------------------
fig1 = bo.plot_acquisition() or plt.gcf()   # Old version returns None -> use gcf
fig1.savefig(f"{OUT_DIR}/acquisition_local.png",
             dpi=150, bbox_inches="tight")
plt.close(fig1)

# ------- (3) Convergence --------------------------------------------------
fig2 = bo.plot_convergence() or plt.gcf()
fig2.savefig(f"{OUT_DIR}/convergence_local.png",
             dpi=150, bbox_inches="tight")
plt.close(fig2)

print("Final kernel parameters:", bo.model.model.kern)
mu, var = bo.model.predict(X_raw)      # Check μ,σ value ranges
print("μ range", mu.min(), mu.max())
print("σ range", np.sqrt(var).min(), np.sqrt(var).max())
print("Suggested point:", bo.suggest_next_locations())

# ===== Get suggested points (already available) =====
next_x = bo.suggest_next_locations()[0]       # shape (2,)
sugg = {"latent": float(next_x[0]), "beta": float(next_x[1])}

# ===== Write to file for BO_flat to read =====
import json, os
with open(os.path.join(OUT_DIR, "next_trial.json"), "w") as f:
    json.dump(sugg, f, indent=2)
