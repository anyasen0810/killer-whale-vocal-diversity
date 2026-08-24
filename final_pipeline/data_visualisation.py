# imports
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr  # matches the Spearman Mantel test used elsewhere in the pipeline
import seaborn as sns

# defining function to load distance matrices
def load_distance_matrix(csv_path, label_order=None):
    df = pd.read_csv(csv_path, index_col=0)
    if label_order is not None:
        df = df.loc[label_order, label_order]
    return df

# defining function to plot heatmap 
def plot_distance_heatmap(dist_matrix, title, cmap, save_path=None,
                           figsize=(6, 5), fmt=".3f"):
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        dist_matrix,
        xticklabels=dist_matrix.columns,
        yticklabels=dist_matrix.index,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Distance"},
        ax=ax,
    )
    ax.set_title(title, pad=12)
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
 
    return fig, ax

def matrix_to_pairs(dist_matrix, value_name):
    
    labels = dist_matrix.index.tolist()
    rows = []
    # using combinations to avoid repeating [airs]
    for a, b in combinations(labels, 2):
        rows.append({
            "pop_a": a,
            "pop_b": b,
            # unique string ID for this pair to align two matrices later
            "pair": f"{a}-{b}",       
            value_name: dist_matrix.loc[a, b],
        })
    return pd.DataFrame(rows)

# defining function to plot MMD versus genetic distance scatterplots 
def plot_pairwise_scatter(x_matrix, y_matrix, x_label, y_label, title,
                           color="steelblue", annotate=True,
                           figsize=(6, 6), save_path=None):
    
    # flattening matrices into pair lists
    x_pairs = matrix_to_pairs(x_matrix, x_label)
    y_pairs = matrix_to_pairs(y_matrix, y_label)

    # merging on the "pair" string to ensure the same subpopulation pair is being compared across matrices
    merged = x_pairs.merge(y_pairs[["pair", y_label]], on="pair")

    # Spearman rank correlation 
    r, p = spearmanr(merged[x_label], merged[y_label])
    
    # creating figure 
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(merged[x_label], merged[y_label], s=70, color=color,
               edgecolor="k", alpha=0.85, zorder=3)

    # labelling points
    if annotate:
        for _, row in merged.iterrows():
            ax.annotate(row["pair"], (row[x_label], row[y_label]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # setting title 
    ax.set_title(f"{title}\n(r = {r:.3f}, n = {len(merged)})")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax, merged, r, p


if __name__ == "__main__":
    # defining order of subpopulations to match matrices 
    LABEL_ORDER = ["AT1", "GAT", "NRKW", "OKW", "SAR", "SRKW", "WCT"]

    # loading matrices 
    mmd = load_distance_matrix("mmd_matrix.csv", label_order=LABEL_ORDER)
    mtdna = load_distance_matrix("mtDNA.csv", label_order=LABEL_ORDER)
    fst = load_distance_matrix("fst_matrix.csv", label_order=LABEL_ORDER)
    
    # plotting mmd heatmap (yellow to brown)
    plot_distance_heatmap(
        mmd,
        title="Acoustic distance (MMD)",
        cmap="YlOrBr",  
        save_path="mmd_heatmap.png",)
 
    # plotting mtdna heatmap (blue-teal)
    plot_distance_heatmap(
        mtdna,
        title="mtDNA patristic distance",
        cmap="BuGn",  
        save_path="mtdna_heatmap.png",)
 
    # plotting nuclear heatmap (teal)
    plot_distance_heatmap(
        fst,
        title="Nuclear differentiation (Fst)",
        cmap="YlGn",  
        save_path="fst_heatmap.png",)
 
    plt.show()

    # plotting MMD vs. mtDNA distance scatterplot
    fig1, ax1, merged1, r1, p1 = plot_pairwise_scatter(
        mmd, mtdna,
        x_label="Acoustic distance (MMD)",
        y_label="mtDNA patristic distance",
        title="Acoustic vs. mtDNA distance (21 subpopulation pairs)",
        color="#e08214",  # warm tone, matches MMD heatmap palette
        save_path="mmd_vs_mtdna_scatter.png",)

    # plotting MMD vs. Fst distance scatterplot 
    fig2, ax2, merged2, r2, p2 = plot_pairwise_scatter(
        mmd, fst,
        x_label="Acoustic distance (MMD)",
        y_label="Nuclear differentiation (Fst)",
        title="Acoustic vs. Fst distance (21 subpopulation pairs)",
        color="#2ca25f",  # teal/green, matches Fst heatmap palette
        save_path="mmd_vs_fst_scatter.png",)

    print("MMD vs mtDNA: r =", round(r1, 3), "p =", round(p1, 3))
    print("MMD vs Fst:   r =", round(r2, 3), "p =", round(p2, 3))