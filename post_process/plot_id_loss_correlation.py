#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Okabe–Ito colorblind-friendly palette
CBLIND = {
    "ESS":   "#0072B2",  # blue
    "TLE":   "#009E73",  # vermillion
    "GRIDE": '#FE6100'
    
}

def pearson_per_layer(log_ids: np.ndarray, loss: np.ndarray) -> np.ndarray:
    num_layers = log_ids.shape[1]
    r = np.empty(num_layers, dtype=float)
    for l in range(num_layers):
        x = log_ids[:, l]
        y = loss
        mask = np.isfinite(x) & np.isfinite(y)
        r[l] = np.nan if mask.sum() < 3 else stats.pearsonr(x[mask], y[mask]).statistic
    return r

def load_model_arrays(root: str, model_name: str, eps: float):
    npz_path = os.path.join(root, "Pile-Structured", model_name, "aggregated_results.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    loss  = data["loss"].astype(float)  # (N,)
    ess   = np.log(np.clip(data["ESS"].astype(float), eps, None))                  # (N, L)
    tle   = np.log(np.clip(data["TLE"].astype(float), eps, None))                  # (N, L)
    gride = np.log(np.clip(data["GRIDE"][..., 0, 1].astype(float), eps, None))     # (N, L) using [:,0,1]
    return {"ESS": ess, "TLE": tle, "GRIDE": gride}, loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results", help="Top-level results dir")
    ap.add_argument("--model", required=True,
                    help="Model name (folder name under Pile-Structured)")
    ap.add_argument("--out", default="figs/id_loss_correlation.png", help="Path to save the plot")
    ap.add_argument("--eps", type=float, default=1e-12, help="Epsilon to avoid log(0)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # plt.figure(figsize=(8, 5))

    # Only show legend labels once (as requested)
    legend_labels = {
        "ESS":   "ESS (k = 10)",
        "TLE":   "TLE (k = 20)",
        "GRIDE": "GRIDE (range scaling = 4)",
    }
    shown_label = {k: False for k in legend_labels.keys()}

    num_layers = None
    model_name = args.model
    ids_dict, loss = load_model_arrays(args.root, model_name, args.eps)
    if num_layers is None:
        num_layers = list(ids_dict.values())[0].shape[1]
    layers = np.arange(1, num_layers + 1)

    for estimator, arr in ids_dict.items():
        r = pearson_per_layer(arr, loss)  # (L,)
        label = legend_labels[estimator] if not shown_label[estimator] else "_nolegend_"
        plt.plot(
            layers,
            r,
            marker=".",
            label=label,
            color=CBLIND[estimator],
            alpha=0.9,
        )
        shown_label[estimator] = True

    # Title with model names
    model_title = {"meta-llama/Meta-Llama-3-8B": "Llama-3-8B"}
    # plt.title(f"ID–Loss Correlation per Layer for {model_title[args.model]}", fontsize="x-large")

    plt.ylabel(r"$\rho(\log ID_{\ell}, \mathrm{surprisal})$", fontsize="x-large")
    plt.xlabel("Layer", fontsize="x-large")
    plt.grid(True, alpha=0.4)
    plt.tick_params(labelsize="x-large")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize="medium")
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved plot to {args.out}")

# Usage: python -m post_process.plot_id_loss_correlation   --root results   --model "meta-llama/Meta-Llama-3-8B"   --out results/figs/id_loss_correlation_llama_estimators.png
if __name__ == "__main__":
    main()
