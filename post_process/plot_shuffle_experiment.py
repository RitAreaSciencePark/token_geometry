#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

# --- your notebook colormap ---
hex6 = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
colors6=[mcolors.to_rgb(i) for i in hex6]
colors = [colors6[0], colors6[3]]
positions = [0, 1]
cmap2 = mcolors.LinearSegmentedColormap.from_list("", list(zip(positions, colors)))

def load_shuffled_npz(root, model):
    path = os.path.join(root, "Pile-Shuffled", model, "aggregated_results.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    data = np.load(path, allow_pickle=True)
    ESS   = data["ESS"].astype(float)                  # (N, L)
    TLE   = data["TLE"].astype(float)                  # (N, L)
    GRIDE = data["GRIDE"][..., 0, 1].astype(float)     # (N, L) -> use [0,1]
    loss  = data["loss"].astype(float)                 # (N,)
    return {"ESS": ESS, "TLE": TLE, "GRIDE": GRIDE}, loss

def infer_prompts(N, num_shuffles):
    if N % num_shuffles != 0:
        raise ValueError(f"N={N} not divisible by num_shuffles={num_shuffles} — "
                         "ordering assumption (rows grouped by prompt) seems broken.")
    return N // num_shuffles

def plot_for_prompt(ids_dict, prompt_idx, num_shuffles, outpath):
    # ids_dict values: (N, L)
    N, L = next(iter(ids_dict.values())).shape
    num_prompts = infer_prompts(N, num_shuffles)
    if not (0 <= prompt_idx < num_prompts):
        raise IndexError(f"prompt_idx={prompt_idx} out of range [0, {num_prompts-1}]")

    base = prompt_idx * num_shuffles
    rows = [base + s for s in range(num_shuffles)]
    layers = np.arange(1, L + 1)

    # --- figure: 2 rows x 3 cols ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8))
    top_axes = axes[0, :]
    bot_axes = axes[1, :]

    estimators = ["ESS", "TLE", "GRIDE"]
    titles = {"ESS": "ESS", "TLE": "TLE", "GRIDE": "GRIDE"}

    # ===== Row 1: the chosen prompt's 0..(num_shuffles-1) curves =====
    for ax, est in zip(top_axes, estimators):
        arr = ids_dict[est]  # (N, L)
        for s, row in enumerate(rows):
            ax.plot(layers, arr[row], marker=".", linewidth=1.2,
                    # label=f"shuffle={s}", 
                    color=cmap2(s/num_shuffles))
        ax.set_title(titles[est])
        ax.set_xlabel("Layer", fontsize="large")
        ax.grid(True, alpha=0.4)
    top_axes[0].set_ylabel("ID", fontsize="large")

    # Colorbar for top row (shuffle index)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=plt.Normalize(vmin=0, vmax=num_shuffles-1))
    sm.set_array([])
    divider = make_axes_locatable(top_axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Shuffle Index', fontsize='x-large')
    cbar.ax.tick_params(labelsize='large')

    # ===== Row 2: mean ± std across ALL prompts for shuffle=0 and shuffle=last =====
    # For each estimator, gather rows: 0::num_shuffles (no shuffle) and (num_shuffles-1)::num_shuffles (full shuffle)
    s0 = 0
    sL = num_shuffles - 1
    for ax, est in zip(bot_axes, estimators):
        arr = ids_dict[est]  # (N, L)

        # stack all prompts for a fixed shuffle slot
        # shape: (num_prompts, L)
        arr_s0 = arr[s0::num_shuffles, :]
        arr_sL = arr[sL::num_shuffles, :]

        # mean ± std across prompts
        mean0 = arr_s0.mean(axis=0); std0 = arr_s0.std(axis=0)
        meanL = arr_sL.mean(axis=0); stdL = arr_sL.std(axis=0)

        # plot
        ax.plot(layers, mean0, marker='.', linewidth=1.8, color=cmap2(s0/num_shuffles), 
                #label=f"shuffle={s0} mean"
                )
        ax.fill_between(layers, mean0-std0, mean0+std0, alpha=0.25, color=cmap2(s0/num_shuffles), linewidth=0)

        ax.plot(layers, meanL, marker='.', linewidth=1.8, color=cmap2(sL/num_shuffles), 
                #label=f"shuffle={sL} mean"
                )
        ax.fill_between(layers, meanL-stdL, meanL+stdL, alpha=0.25, color=cmap2(sL/num_shuffles), linewidth=0)

        ax.set_title(titles[est])
        ax.set_xlabel("Layer", fontsize="large")
        ax.grid(True, alpha=0.4)
        # ax.legend(fontsize="small", loc="best")
    bot_axes[0].set_ylabel("ID", fontsize="large")
    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=plt.Normalize(vmin=0, vmax=num_shuffles-1))
    sm2.set_array([])
    divider2 = make_axes_locatable(bot_axes[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(sm2, cax=cax2)
    cbar2.set_label('Shuffle Index', fontsize='x-large')
    cbar2.ax.tick_params(labelsize='large')
    fig.suptitle(f"Per-layer ID across shuffle degrees - prompt {np.load('subset_indices.npy')[prompt_idx]}\nBottom row: mean±std over 50 prompts for unshuffled and shuffled prompts",
                 fontsize="x-large")

    # plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    print(f"✅ Saved: {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results", help="Top-level results dir")
    ap.add_argument("--model", required=True, help="Model folder under Pile-Shuffled")
    ap.add_argument("--prompt_idx", type=int, required=True, help="Prompt index (0-based)")
    ap.add_argument("--num_shuffles", type=int, default=6, help="Number of shuffle degrees per prompt")
    ap.add_argument("--out", default="results/figs/shuffled_id_across_estimators.png", help="Output PNG")
    args = ap.parse_args()

    ids_dict, _ = load_shuffled_npz(args.root, args.model)
    plot_for_prompt(ids_dict, args.prompt_idx, args.num_shuffles, args.out)

# Usage: python -m post_process.plot_shuffle_experiment   --root results   --model "meta-llama/Meta-Llama-3-8B"   --prompt_idx 14 
if __name__ == "__main__":
    main()
