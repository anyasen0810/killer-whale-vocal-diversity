# imports
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import pandas as pd
from pathlib import Path
import os

# setting up directories
data_root = Path("subpopulation_embeddings_v2")
csv_files = sorted(data_root.glob("*_embeddings.csv"))
base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
output_mmd = os.path.join(base_dir, "mmd_matrix.csv")

# defining function that calculates MMD between two distributions 
def calculate_single_mmd(X, Y, gamma = None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
        
    # calculating initial kernel matrices
    k_XX = rbf_kernel(X, X, gamma=gamma)
    k_YY = rbf_kernel(Y, Y, gamma=gamma)
    k_XY = rbf_kernel(X, Y, gamma=gamma)
    sp1_size = X.shape[0]
    sp2_size = Y.shape[0]

    # normalising similarity
    n_XX = np.sum(k_XX) / (sp1_size ** 2)
    n_YY = np.sum(k_YY) / (sp2_size ** 2)
    n_XY = np.sum(k_XY) / (sp1_size * sp2_size)

    # calculating MMD as within-group similarity - between-group similarity 
    return n_XX + n_YY - 2 * n_XY

# loading embeddings from embeddings csv files 
acoustic_data = {}
for csv_file in csv_files:
    # getting subpop label from csv file name 
    subpop = csv_file.stem.replace("_embeddings", "")
    df = pd.read_csv(csv_file)
    features_only = df.select_dtypes(include=[np.number]).to_numpy()
    acoustic_data[subpop] = features_only
    print(f"{subpop} data loaded. Shape of feature matrix is {features_only.shape}")

subpops = list(acoustic_data.keys())
n_subpops = len(subpops)

# creating empty matrix for MMD scores
mmd_matrix = np.zeros((n_subpops, n_subpops))

for i in range(n_subpops):
    for j in range(i, n_subpops):
        # taking distance of subpopulation from itself as 0 
        if i == j:
            mmd_matrix[i, j] = 0.0
        else:
            dist = calculate_single_mmd(acoustic_data[subpops[i]], acoustic_data[subpops[j]])
            mmd_matrix[i, j] = dist
            mmd_matrix[j, i] = dist

df_acoustic_mmd = pd.DataFrame(mmd_matrix, index=subpops, columns=subpops)
print(df_acoustic_mmd)

# saving dataframe to csv 
df_acoustic_mmd.to_csv(output_mmd)
print(f"Saved to {output_mmd}")

