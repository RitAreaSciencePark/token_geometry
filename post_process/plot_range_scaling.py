import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
# from natsort import natsorted


hex6 = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
colors6=[mcolors.to_rgb(i) for i in hex6]
colors = [colors6[0], colors6[3]]
positions = [0, 1]
cmap2 = mcolors.LinearSegmentedColormap.from_list("", list(zip(positions, colors)))

# Note: Cosine similarity is defined for the embedding layers whereas ID and NO are defined for the model layers.

all_models = ['meta-llama/Meta-Llama-3-8B']
NCOL = 3 if len(all_models) <= 3 else 2
model_titles = {'meta-llama/Meta-Llama-3-8B' : 'Llama-3-8B'}
num_prompts, num_layers = 2244, 32
input_dir = 'results/Pile-Shuffled'
losses = {}
"""
Older version
for model_name in all_models:
    ifiles = f'{input_dir}/{model_name}/losses/*.npy'
    filelst= natsorted(glob.glob(ifiles)) 
    losses[model_name] = np.concatenate([np.load(file) for file in filelst], axis = 0)
    np.save(f'{input_dir}/{model_name}/summaries/losses.npy', losses[model_name])
"""
# for model_name in all_models:
#     losses[model_name] = np.load(f'{input_dir}/{model_name}/summaries/losses.npy')
    
# csn_sims = {}
# for model_name in all_models:
#     csn_sims[model_name] = np.load(f'{input_dir}/{model_name}/summaries/csn_sim.npy').transpose([0, 2, 1])
    
ids = {}
for model_name in all_models:
    ids[model_name] = np.load(f'{input_dir}/{model_name}/aggregated_results.npz')['GRIDE'][:,:, 0, :]
    
# mnos = {}
# for model_name in all_models:
#     mnos[model_name] = np.load(f'{input_dir}/{model_name}/summaries/mnos.npy')

# nn_sims = {}
# for model_name in all_models:
#     if model_name == 'Llama-3-8B':
#         nn_sims[model_name] = np.load(f'{input_dir}/{model_name}/summaries/nn_sim.npy')    
# cmp = {'Llama-3-8B': colors6[3], 'Mistral-7B': colors6[4], 'Pythia-6.9B': colors6[0], 'Pythia-6.9B-Deduped': colors6[1]}

cmp = {'Llama-3-8B': colors6[3], 'Mistral-7B': colors6[2], 'Pythia-6.9B': colors6[0]}

TEST_INDX_SHUFFLE = 14 # The test prompt we use throughout the paper to display results for a single prompt.
filtered_indices = np.load('../../filtered_indices.npy')
subset_indices = np.load('../../subset_indices.npy')
TEST_INDX = np.where(filtered_indices == subset_indices[TEST_INDX_SHUFFLE])[0][0] # The test prompt we use throughout the paper to display results for a single prompt.
TEST_INDX

num_layers =32

start_ind = 1
end_ind = num_layers + 1
xrange = np.arange(start_ind, end_ind)
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey="row")
rs_idx = 1
model_name = all_models[0]
NUM_SHUFFLES = 6
for jdx in range(2):
    ax = axes[jdx]
    ops = ids[model_name][:, :, rs_idx]
    if jdx == 0: ax.set_ylabel(f"ID", fontsize="x-large")
    if jdx == 0:
        for qdx, shuffle_idx in enumerate(range(NUM_SHUFFLES * TEST_INDX_SHUFFLE, NUM_SHUFFLES * (TEST_INDX_SHUFFLE + 1))):
            ax.plot(xrange, ops[shuffle_idx], marker='.', c=cmap2(qdx/NUM_SHUFFLES))
    else:
        
        for qdx in [0, 5]:
            op = ops[qdx::6, :]

            # Calculate mean and standard deviation
            mean_op = op.mean(axis=0)
            std_op = op.std(axis=0)

            # Plot the mean curve
            ax.plot(xrange, mean_op, label=model_name, c=cmap2(qdx/NUM_SHUFFLES), marker='.')

            # Fill between the mean +/- std (shaded area)
            ax.fill_between(np.arange(start_ind, end_ind), mean_op - std_op, mean_op + std_op, color=cmap2(qdx/NUM_SHUFFLES), alpha=0.2)

    ax.set_xlabel("Layer", fontsize="x-large")
    ax.tick_params(which='both', labelsize="x-large")
    ax.grid(True)

# Create a ScalarMappable object for the color bar
sm = plt.cm.ScalarMappable(cmap=cmap2, norm=plt.Normalize(vmin=0, vmax=NUM_SHUFFLES-1))
sm.set_array([])

# Add color bar
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('Shuffle Index', fontsize='x-large')
cbar.ax.tick_params(labelsize='x-large')

plt.tight_layout()
handles_psim, labels_psim = axes[0].get_legend_handles_labels()
plt.savefig(f'results/figs/ids_llama_{TEST_INDX_SHUFFLE}_scaling_{2**(rs_idx + 1)}.png', bbox_inches='tight', dpi=300)
plt.show()

