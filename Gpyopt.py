# post_plot_gpyopt.py
import numpy as np
if not hasattr(np, "bool"):        # NumPy 1.24+ 拿掉了 np.bool
    np.bool = bool    
import pandas as pd
import matplotlib
matplotlib.use("Agg")
setattr(matplotlib, "numpy", np)      # 給 GPy 找得到 matplotlib.numpy
import matplotlib.pyplot as plt
import GPy
from GPyOpt.methods import BayesianOptimization as BO
OUT_DIR = "change_seed"         # 放 CSV 的資料夾

df = pd.read_csv(f"{OUT_DIR}/all_trials.csv")
# 只保留成功的 trial
df = df[df["value"].notna()].reset_index(drop=True)

# 構成 X, Y 供 GPyOpt 視覺化
X_raw = df[["params_latent","params_beta"]].to_numpy()
Y_raw = -df["value"].to_numpy().reshape(-1,1)    # ← 這裡通常做負號


# ==== 1. 重新處理 Y：min-max 而非 z-score =========
y_min, y_max = Y_raw.min(), Y_raw.max()
Y = (Y_raw - y_min) / (y_max - y_min)      # 0 ~ 1, 越小越差

# ==== 2. Kernel：Matern52 + Bias + White ===========
k_main = GPy.kern.Matern52(
    input_dim=2, ARD=True,
    variance=2.0,                 # 讓 μ 對 y 幅度更敏感
    lengthscale=[0.2, 0.3]        # x1, x2 的初值
)
k_main.lengthscale.constrain_bounded(0.2, 3)    # 避免竹簾 or 拉平
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
    {"name": "latent", "type": "continuous", "domain": (5, 25)},
    {"name": "beta",   "type": "continuous", "domain": (0.0, 2.0)}
]

# bo = BO(f=None, domain=domain, X=X, Y=Y, normalize_Y=False)
bo = BO(
    f=None, domain=domain,
    X=X_raw, Y=Y,
    kernel=kernel,
    normalize_Y=False,     # ← y 已 min-max
    exact_feval=True,
    # acquisition_type='LCB',
    # acquisition_weight=6.0,
    acquisition_type='EI',
    acquisition_par=0.15,
    optimizer_restarts=20
)
# bo.acquisition_type  = 'EI'
# bo.acquisition_par   = 0.15
bo._compute_results()                # 只做 GP 擬合，不再跑函數

# ------- (2) Acquisition + posterior -------------------------------------
fig1 = bo.plot_acquisition() or plt.gcf()   # 舊版回傳 None -> 用 gcf
fig1.savefig(f"{OUT_DIR}/acquisition_local.png",
             dpi=150, bbox_inches="tight")
plt.close(fig1)

# ------- (3) Convergence --------------------------------------------------
fig2 = bo.plot_convergence() or plt.gcf()
fig2.savefig(f"{OUT_DIR}/convergence_local.png",
             dpi=150, bbox_inches="tight")
plt.close(fig2)

print("kernel 最終參數：", bo.model.model.kern)
mu, var = bo.model.predict(X_raw)      # 看 μ,σ 數值範圍
print("μ range", mu.min(), mu.max())
print("σ range", np.sqrt(var).min(), np.sqrt(var).max())
print("建議點:", bo.suggest_next_locations())

# ===== 取得建議點 (已經有了) =====
next_x = bo.suggest_next_locations()[0]       # shape (2,)
sugg = {"latent": float(next_x[0]), "beta": float(next_x[1])}

# ===== 寫到檔案，給 BO_flat 讀 =====
import json, os
with open(os.path.join(OUT_DIR, "next_trial.json"), "w") as f:
    json.dump(sugg, f, indent=2)
print("✔ 建議點已寫入 next_trial.json :", sugg)
