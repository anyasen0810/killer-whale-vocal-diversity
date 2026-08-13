import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd

# defining function that calculates MMD between two distributions 
def calculate_single_mmd(X,Y, gamma = None):
    if gamma is None:
        gamma = 1.0/X.shape[1] # scaling bandwidth with size of vector 

    # calculate initial kernal matrices 
    # matrix for internal similarity in first subpopulation 
    k_XX = rbf_kernel(X, X, gamma = gamma)
    # matrix for internal similarity in second subpopulation
    k_YY = rbf_kernel(Y, Y, gamma = gamma)
    # matrix for similarity between the two subpopulations 
    k_XY = rbf_kernel(X, Y, gamma = gamma)

    # obtaining number of samples in each subpopulation 
    # number of samples in first subpopulation
    sp1_size = X.shape[0]
    sp2_size = Y.shape[0]

    # normalising similarity 
    n_XX = np.sum(k_XX) / (sp1_size * sp2_size)
    n_YY = np.sum(k_YY) / (sp1_size * sp2_size)
    n_XY = np.sum(k_XY) / (sp1_size * sp2_size)

    # returning MMD distance between the subpopulations
    return n_XX + n_YY - 2 * n_XY   

# loading embeddings data
embeddings_df = pd.read_csv("features.csv") # double check name 

# identifying subpopulation column 
subpop_column = "Ecotype"

# getting list of all subpops present in the data
subpops = embeddings_df[subpop_column].unique()
# getting number of subpopulations 
n_subpops = len(subpops)

acoustic_data = {}

for subpop in subpops:
    # identifying rows belonging to single subpopulation 
    subpop_rows = embeddings_df[embeddings_df[subpop_column] == subpop]

    features_only = subpop_rows.drop(columns = [subpop_column]).to_numpy

    acoustic_data[subpop] = features_only
    print(f"{subpop} data loaded. Shape of feature matrix is {features_only.shape}")

# creating empty matrix for MMD scores
mmd_matrix = np.zeros((n_subpops, n_subpops))

# using MMD function to fill in empty matrix
for i in range(n_subpops):
    for j in range(i, n_subpops):
        # skipping mmd calculation if subpop is being compared against itself 
        if i == j:
            mmd_matrix[i, j] = 0.0
        else:
            # calculate_mmd is the function from our previous step
            dist = calculate_single_mmd(acoustic_data[subpops[i]], acoustic_data[subpops[j]])
            mmd_matrix[i, j] = dist
            # mirroring the distance to prevent mmd being calculated twice for same pair of subpops
            mmd_matrix[j, i] = dist  

df_acoustic_mmd = pd.DataFrame(mmd_matrix, index=subpops, columns=subpops)
print(df_acoustic_mmd)