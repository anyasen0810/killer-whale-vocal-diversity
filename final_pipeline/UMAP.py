# imports 
import pandas as pd
import glob
import os
import numpy as np
import umap
import matplotlib.pyplot as plt

# setting up directories 
base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
embeddings_dir = os.path.join(base_dir, "subpopulation_embeddings_v2")
output_dir = os.path.join(base_dir, "umap_results")
os.makedirs(output_dir, exist_ok=True)

# finding all subpopulation embedding csvs
embedding_files = glob.glob(os.path.join(embeddings_dir, "*_embeddings.csv"))
print(f"Found {len(embedding_files)} subpopulation embedding files")

# loading each file and tagging rows with their subpopulation
all_dfs = []
for file_path in embedding_files:
    subpop_name = os.path.basename(file_path).replace("_embeddings.csv", "")
    df = pd.read_csv(file_path)
    df["Subpopulation"] = subpop_name
    all_dfs.append(df)
    print(f"  {subpop_name}: {len(df)} clips loaded")

# combining embeddings files for each subpopulation 
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal combined: {len(combined_df)} clips across {combined_df['Subpopulation'].nunique()} subpopulations")

# separating out the embedding vector columns from the metadata columns
vector_cols = [c for c in combined_df.columns if c.startswith("vector")]
embedding_matrix = combined_df[vector_cols].values
print(f"Embedding matrix shape: {embedding_matrix.shape}")

# running UMAP
reducer = umap.UMAP(
    # number of neighbours set to scale with size of dataset
    n_neighbors= max(5, min(100, int(0.001 * combined_df.shape[0]))),  
    min_dist=0.1,
    # projecting into 2d space 
    n_components=2,
    # choosing euclidean as that's what will be used for distance matrices 
    metric='euclidean',     
    random_state=42      
)
umap_embedding = reducer.fit_transform(embedding_matrix)

# saving values from each dimension as a column in the dataframe 
combined_df["UMAP1"] = umap_embedding[:, 0]
combined_df["UMAP2"] = umap_embedding[:, 1]

# saving the UMAP coordinates alongside metadata for later use 
umap_output_file = os.path.join(output_dir, "umap_coordinates.csv")
combined_df[["Filename", "Subpopulation", "UMAP1", "UMAP2"]].to_csv(umap_output_file, index=False)
print(f"Saved UMAP coordinates to {umap_output_file}")

# plotting umap coorindates 
plt.figure(figsize=(10, 8))
subpopulations = combined_df["Subpopulation"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(subpopulations)))

# visualising results as scatterplot 
# differentiating between subpopulations by colour 
for subpop, color in zip(subpopulations, colors):
    mask = combined_df["Subpopulation"] == subpop
    plt.scatter(
        combined_df.loc[mask, "UMAP1"],
        combined_df.loc[mask, "UMAP2"],
        label=subpop,
        color=color,
        alpha=0.7,
        s=20)

# labelling axes 
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Projection of Perch Embeddings by Subpopulation")
plt.legend(title="Subpopulation", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# saving plot as png 
plot_file = os.path.join(output_dir, "umap_plot.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Saved plot to {plot_file}")
plt.show()

