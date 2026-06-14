#!/usr/bin/env python3
"""Generate every figure and number used in ../analysis.md, in one run.

Produces three figures in ../figs:
  1. box_entropy_vs_dimension.png  -- Monte-Carlo entropy of the U[0,L]^D box model
                                      (no model needed).
  2. llama_box_width_scatter.png   -- per-prompt entropy vs logit-box length on
                                      Llama-3-8B, 50 Pile prompts.
  3. llama_logit_spectrum.png      -- gauge-invariant logit histogram and sorted
                                      logit spectrum for a single prompt.

and prints / saves (../outputs/summary.json) the capture table and box-length stats.

Llama is loaded once; the 50-prompt sweep and the single-prompt spectrum share that
load (the spectrum reuses the first prompt's forward pass).

Inputs (all already in the repo): the GRIDE results in
results/Pile-Structured/meta-llama/Meta-Llama-3-8B (logits_id.npy, contextual_entropy.npy)
and the repo-root filtered_indices.npy / subset_indices.npy; plus the cached
meta-llama/Meta-Llama-3-8B weights and the NeelNanda/pile-10k dataset (offline, GPU).
"""
import os
import json
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/token_geometry_matplotlib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve()
ANALYSIS = HERE.parents[1]
REPO = HERE.parents[3]
RESULTS = REPO / "results" / "Pile-Structured" / "meta-llama" / "Meta-Llama-3-8B"
FIGS = ANALYSIS / "figs"
OUT = ANALYSIS / "outputs"

MODEL = "meta-llama/Meta-Llama-3-8B"
MAXLEN = 1024
KMAX = 128                      # capture/box curves out to here (>> max per-prompt D_p)

# single-prompt spectrum settings
SPECTRUM_PROMPT = 0             # which subset prompt (0 = first)
N_POS = 6                       # token positions to overlay
RANK_LOG = 4096                 # log-rank depth for the spectrum panel


# ---------------------------------------------------------------- figure 1: box model
def entropy_from_logits(z):
    z_max = np.max(z, axis=1, keepdims=True)
    exp_shifted = np.exp(z - z_max)
    partition = np.sum(exp_shifted, axis=1, keepdims=True)
    probs = exp_shifted / partition
    log_partition = z_max[:, 0] + np.log(partition[:, 0])
    return log_partition - np.sum(probs * z, axis=1)


def box_entropy_curves(d_values, l_values, n_samples=20_000, seed=1234):
    rng = np.random.default_rng(seed)
    means = np.empty((len(l_values), len(d_values)))
    stderr = np.empty_like(means)
    for col, d in enumerate(d_values):
        base = rng.uniform(0.0, 1.0, size=(n_samples, int(d)))
        for row, L in enumerate(l_values):
            if L == 0:
                means[row, col], stderr[row, col] = 1.0, 0.0
                continue
            ent = entropy_from_logits(L * base)
            means[row, col] = ent.mean() / np.log(d)
            stderr[row, col] = ent.std(ddof=1) / np.sqrt(n_samples) / np.log(d)
    return means, stderr


def make_box_entropy_figure():
    d_values = np.arange(2, 41)
    l_values = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
    means, stderr = box_entropy_curves(d_values, l_values)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(l_values)))
    for row, (L, color) in enumerate(zip(l_values, colors)):
        ax.plot(d_values, means[row], color=color, linewidth=2.0, label=f"L = {L:g}")
        ax.fill_between(d_values, means[row] - 2 * stderr[row], means[row] + 2 * stderr[row],
                        color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Box dimension D")
    ax.set_ylabel(r"$\mathbb{E}[S]/\log D$")
    ax.set_title(r"Normalized entropy for $\mathcal{U}[0,L]^D$ logits")
    ax.set_xlim(d_values[0], d_values[-1]); ax.set_ylim(0.0, 1.03)
    ax.set_xticks([2, 10, 20, 30, 40]); ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, title="Logit scale", loc="center left",
              bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.savefig(FIGS / "box_entropy_vs_dimension.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"max normalized standard error: {np.max(stderr):.5f}", flush=True)


# ---------------------------------------------------------------- Llama sweep + spectrum
def load_llama():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers.modeling_utils as _mu
    if getattr(_mu, "ALL_PARALLEL_STYLES", None) is None:
        _mu.ALL_PARALLEL_STYLES = ["colwise", "rowwise", "colwise_rep", "rowwise_rep",
                                   "local_colwise", "local_rowwise", "local", "gather",
                                   "local_packed_rowwise", "sequence_parallel", "replicate"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.base_model_tp_plan = None
    model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, torch_dtype=torch.float32).to("cuda")
    return model.eval(), tok


def run_llama():
    import torch
    from datasets import load_dataset
    torch.set_grad_enabled(False)

    fi = np.load(REPO / "filtered_indices.npy")
    # D_p = GRIDE intrinsic dimension of each prompt's per-token logit cloud, taken from
    # the committed GRIDE results (range-scaling 0, ID column); per-prompt mean entropy
    # from the same directory. Both are row-aligned to filtered_indices.
    logit_id = np.load(RESULTS / "logits_id.npy")[:, 0, 1]   # D_p per prompt
    cent = np.load(RESULTS / "contextual_entropy.npy")
    subset = np.load(REPO / "subset_indices.npy")
    sel = np.array([int(np.where(fi == v)[0][0]) for v in subset])

    seqs = load_dataset("NeelNanda/pile-10k")["train"]["text"]
    model, tok = load_llama()

    ks = np.arange(1, KMAX + 1)
    sum_Sc = np.zeros(KMAX); sum_M = np.zeros(KMAX); sum_W = np.zeros(KMAX)
    sum_Sfull = 0.0; tot_tok = 0
    all_Wkid = []
    rows = []
    spectrum = None

    for n, i in enumerate(sel):
        pile = int(fi[i])
        k_id = int(max(1, min(KMAX, round(float(logit_id[i])))))
        enc = tok(seqs[pile].strip(), add_special_tokens=False, return_tensors="pt",
                  max_length=MAXLEN, truncation=True).to("cuda")
        logits = model(**enc).logits[0].float()             # (T, V)
        logp = torch.log_softmax(logits, dim=-1)
        Sfull = -(logp.exp() * logp).sum(-1)                # (T,)
        tlp, _ = torch.topk(logp, KMAX, dim=-1)
        tp = tlp.exp()
        Sc = torch.cumsum(-(tp * tlp), dim=-1)
        M = torch.cumsum(tp, dim=-1)
        z, _ = torch.topk(logits, KMAX, dim=-1)
        W = z[:, :1] - z

        T = int(Sfull.shape[0])
        sum_Sc += Sc.sum(0).double().cpu().numpy()
        sum_M += M.sum(0).double().cpu().numpy()
        sum_W += W.sum(0).double().cpu().numpy()
        sum_Sfull += float(Sfull.sum()); tot_tok += T

        Sc_kid, M_kid, W_kid = Sc[:, k_id - 1], M[:, k_id - 1], W[:, k_id - 1]
        all_Wkid.append(W_kid.cpu().numpy())
        rows.append(dict(pile=pile, k_id=k_id, id_raw=float(logit_id[i]), n_tok=T,
                         meanS_full=float(Sfull.mean()), cent_saved=float(cent[i]),
                         sumSc_kid=float(Sc_kid.sum()), sumSfull=float(Sfull.sum()),
                         captured_entropy_frac=float(Sc_kid.sum()) / float(Sfull.sum()),
                         captured_mass=float(M_kid.mean()), mean_boxlen=float(W_kid.mean())))

        # reuse this forward pass for the single-prompt spectrum figure
        if n == SPECTRUM_PROMPT:
            pos = np.linspace(int(0.1 * T), T - 1, N_POS).round().astype(int)
            zpos = logits[pos]
            gap = (zpos.max(dim=-1, keepdim=True).values - zpos).reshape(-1)
            gap = gap[gap > 0].cpu().numpy().astype(np.float32)
            Klog = min(RANK_LOG, int(logits.shape[1]))
            zsort, _ = torch.topk(zpos, Klog, dim=-1)
            spectrum = dict(pile=pile, D_p=k_id, pos=pos, ranks=np.arange(1, Klog + 1),
                            zsort=zsort.cpu().numpy(), gap_flat=gap)

        print(f"[{n+1}/{len(sel)}] pile {pile}: k=D_p={k_id:2d}  T={T}  "
              f"<S>={float(Sfull.mean()):.3f}  mass@D_p={float(M_kid.mean()):.3f}  "
              f"entfrac@D_p={float(Sc_kid.sum())/float(Sfull.sum()):.3f}  "
              f"boxlen@D_p={float(W_kid.mean()):.3f}", flush=True)
        del logits, logp, Sfull, tlp, tp, Sc, M, z, W
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    all_Wkid = np.concatenate(all_Wkid)
    mean_M, mean_frac = sum_M / tot_tok, (sum_Sc / tot_tok) / (sum_Sfull / tot_tok)
    summary = dict(
        n_prompts=int(len(df)), n_tokens=int(tot_tok),
        mean_k_id=float(df.k_id.mean()), median_k_id=int(df.k_id.median()),
        mean_S_full=float(sum_Sfull / tot_tok), exp_mean_S_full=float(np.exp(sum_Sfull / tot_tok)),
        captured_mass_at_Dp=float((df.captured_mass * df.n_tok).sum() / df.n_tok.sum()),
        captured_entropy_frac_at_Dp=float(df.sumSc_kid.sum() / df.sumSfull.sum()),
        mass_at_k={str(k): float(mean_M[k - 1]) for k in [1, 3, 16, 32, 128]},
        entropy_frac_at_k={str(k): float(mean_frac[k - 1]) for k in [1, 3, 16, 32, 128]},
        boxlen_at_Dp_mean=float(all_Wkid.mean()), boxlen_at_Dp_median=float(np.median(all_Wkid)),
        rho_S_boxlen=float(stats.pearsonr(df.meanS_full, df.mean_boxlen).statistic),
        rho_S_boxlen_spearman=float(stats.spearmanr(df.meanS_full, df.mean_boxlen).statistic),
        rho_Dp_boxlen=float(stats.pearsonr(df.id_raw, df.mean_boxlen).statistic),
    )
    return df, summary, spectrum


def make_scatter_figure(df, summary):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(df.meanS_full, df.mean_boxlen, s=26, alpha=.65, color="#1f8a70")
    ax.set_xlabel(r"per-prompt mean entropy  $\langle S\rangle_p$")
    ax.set_ylabel(r"per-prompt box length  $W_p$  (units)")
    ax.set_title(f"Llama-3-8B: entropy vs box length  "
                 f"(rho={summary['rho_S_boxlen_spearman']:+.2f}, n={len(df)})")
    ax.grid(True, alpha=.25)
    fig.tight_layout(); fig.savefig(FIGS / "llama_box_width_scatter.png", dpi=130)
    plt.close(fig)


def make_spectrum_figure(sp):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sp["pos"])))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

    bins = np.linspace(0.0, float(sp["gap_flat"].max()), 70)
    ax[0].hist(sp["gap_flat"], bins=bins, color="#1f8a70", alpha=0.85)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"logit gap below top  $z_{(1)} - z_i$")
    ax[0].set_ylabel("number of vocabulary tokens")
    ax[0].set_title(f"Gauge-invariant logit histogram ({len(sp['pos'])} token positions)")

    for r, c in zip(range(len(sp["pos"])), colors):
        ax[1].plot(sp["ranks"], sp["zsort"][r], color=c, lw=1.6, label=f"token {int(sp['pos'][r])}")
    ax[1].axvline(sp["D_p"], ls="--", color="#d62728", label=f"$D_p={sp['D_p']}$")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("rank $r$ (log)"); ax[1].set_ylabel("sorted logit $z_{(r)}$")
    ax[1].set_title("Sorted logit spectrum (log rank)")
    ax[1].set_xlim(1, sp["ranks"][-1]); ax[1].legend(fontsize=7, ncol=2)

    fig.suptitle(f"Llama-3-8B logit spectrum, single prompt "
                 f"(pile {sp['pile']}, $D_p={sp['D_p']}$)")
    fig.tight_layout(); fig.savefig(FIGS / "llama_logit_spectrum.png", dpi=130)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    print("=== figure 1: box-model Monte Carlo ===", flush=True)
    make_box_entropy_figure()

    print("=== figures 2-3: Llama-3-8B ===", flush=True)
    df, summary, spectrum = run_llama()
    make_scatter_figure(df, summary)
    make_spectrum_figure(spectrum)

    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print("\nSUMMARY", json.dumps(summary, indent=2), flush=True)
    print(f"\nDONE -> figures in {FIGS}, numbers in {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
