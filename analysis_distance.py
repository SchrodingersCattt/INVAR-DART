import numpy as np
import matplotlib.pyplot as plt
# Deleted:from sklearn.decomposition import PCA
# Deleted:from sklearn.manifold import TSNE
import pandas as pd
from stat_distribution import ExcelParser
from tqdm import tqdm
# Deleted:import umap

import numpy as np
import matplotlib.pyplot as plt
import umap

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# —— Composition data tools —— #
def closure(x, axis=1):
    s = np.sum(x, axis=axis, keepdims=True)
    s[s==0] = 1.0
    return x / s

def clr_transform(x, eps=1e-6):
    # x: already closed (sum=1). Add pseudocount to handle zeros.
    x_safe = np.clip(x, eps, None)
    gm = np.exp(np.mean(np.log(x_safe), axis=1, keepdims=True))
    return np.log(x_safe / gm)

# —— Fixed functions —— #
def extract_composition_data(parser):
    Fe, Ni, Co, Cr, V, Cu = [], [], [], [], [], []
    tec = []
    for row in parser.iterate_rows():
        tec.append(parser.get_tec(row))
        Fe.append(parser.get_Fe(row))
        Ni.append(parser.get_Ni(row))
        Co.append(parser.get_Co(row))
        Cr.append(parser.get_Cr(row))
        V.append(parser.get_V(row))
        Cu.append(parser.get_Cu(row))

    # Original 6 elements mapped to 8-element order: [Al, Cr, Fe, Co, Ni, Ti, V, Cu]
    comps6 = np.column_stack((Fe, Cr, Co, Ni, V, Cu)).astype(float)

    # First closure (sum to 1)
    comps6 = closure(comps6)

    # Add Al and Ti columns as 0, then do closure again (strictly ensure sum=1)
    zeros = np.zeros((comps6.shape[0], 1))
    # Note order: Al, Cr, Fe, Co, Ni, Ti, V, Cu
    compositions = np.column_stack((zeros,           # Al
                                    comps6[:, 1],    # Cr
                                    comps6[:, 0],    # Fe
                                    comps6[:, 2],    # Co
                                    comps6[:, 3],    # Ni
                                    zeros,           # Ti
                                    comps6[:, 4],    # V
                                    comps6[:, 5]))   # Cu
    compositions = closure(compositions)  # Close again
    tec = np.array(tec, dtype=float)
    return compositions, tec

def visualize_compositions_tsne(compositions_clr, tec):
    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_data = reducer.fit_transform(compositions_clr)
    plt.figure(figsize=(6,6))
    sc = plt.scatter(reduced_data[:,0], reduced_data[:,1], c=tec, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='TEC')
    plt.xlabel('TSNE-1'); plt.ylabel('TSNE-2'); plt.title('TSNE (on CLR space)')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    return plt.gcf(), reducer, reduced_data

def main():
    # Read data
    data_file = "dataset/Data_base_DFT_Thermal.xlsx"
    parser = ExcelParser(data_file)
    compositions, tec = extract_composition_data(parser)  # Already 8-dim, closed to 1

    print(f"Loaded {len(compositions)} compositions with {compositions.shape[1]} elements each")

    # Original 3 compositions (order strictly [Al, Cr, Fe, Co, Ni, Ti, V, Cu])
    original_compositions = np.array([
        [2.44, 5.69, 54.82, 15.49, 21.54, 0,    0, 0],
        [0,    1.02, 57.36, 10.08, 27.32, 4.22, 0, 0],
        [1.32, 1.13, 61.61, 6.59,  29.35, 0,    0, 0],
    ], dtype=float)
    
    # New 3 compositions (order strictly [Al, Cr, Fe, Co, Ni, Ti, V, Cu])
    new_compositions = np.array([
        [0,    0.0778, 0.501, 0.2072, 0.198, 0.0059, 0.0102, 0],  # new_1 - 1st group
        [0,    0.0545, 0.584, 0.1166, 0.2402, 0.0046, 0, 0],      # new_1 - 2nd group
        [0.0009, 0.0978, 0.4052, 0.4476, 0.0416, 0.0029, 0.0041, 0]  # new_1 - 3rd group
    ], dtype=float)

    # Combine all compositions
    all_new_compositions = np.vstack([original_compositions, new_compositions])
    
    # Normalize to ratios: divide by sum (closure)
    all_new_compositions = closure(all_new_compositions)

    # —— Optional: if not doing CLR, at least change UMAP's metric to 'cosine' —— #
    # reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')

    # —— Recommended: do UMAP in CLR space —— #
    compositions_clr = clr_transform(compositions)
    new_clr = clr_transform(all_new_compositions)

    fig, reducer, reduced_data = visualize_compositions_tsne(compositions_clr, tec)

    # Correct: old points use reduced_data, new points use transform result
    # Note: TSNE does not support transform, so we need to fit on combined data
    all_data_clr = np.vstack([compositions_clr, new_clr])
    all_reduced = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_data_clr)
    
    # Separate the reduced data
    reduced_data = all_reduced[:-6]  # Original data
    all_new_reduced = all_reduced[-6:]   # All new compositions (6 compositions)
    
    # Separate into original (new_0) and new (new_1)
    new_0_reduced = all_new_reduced[:3]  # First 3 are new_0
    new_1_reduced = all_new_reduced[3:]  # Last 3 are new_1

    # Overlay plot
    plt.figure(figsize=(6,6))
    sc = plt.scatter(reduced_data[:,0], reduced_data[:,1], c=tec, cmap='viridis', alpha=0.7, label='Existing')
    plt.colorbar(sc, label='TEC')
    
    # Plot new_0 compositions
    markers_0 = ['o','s','^']
    colors_0 = ['red','orange','magenta']
    for i in range(new_0_reduced.shape[0]):
        plt.scatter(new_0_reduced[i,0], new_0_reduced[i,1], marker=markers_0[i], s=80, linewidths=1, color=colors_0[i], label=f'new_0_{i}', edgecolors='black')
    
    # Plot new_1 compositions
    markers_1 = ['D','v','p']
    colors_1 = ['cyan','lime','yellow']
    for i in range(new_1_reduced.shape[0]):
        plt.scatter(new_1_reduced[i,0], new_1_reduced[i,1], marker=markers_1[i], s=80, linewidths=1, color=colors_1[i], label=f'new_1_{i}', edgecolors='black')
    
    plt.xlabel('TSNE-1'); plt.ylabel('TSNE-2'); plt.title('TSNE (CLR) with New Compositions')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("composition_tsne_withnew.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()