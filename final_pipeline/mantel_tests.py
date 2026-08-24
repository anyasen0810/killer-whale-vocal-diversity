# imports 
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# defining mantel test function 
def mantel_test(matrix1, matrix2, method="spearman", n_permutations=9999, random_state=None):
    
    rng = np.random.default_rng(random_state)

    labels1 = list(matrix1.index)
    labels2 = list(matrix2.index)
    if labels1 != labels2:
        raise ValueError(
            f"Matrix labels/order don't match.\nmatrix1: {labels1}\nmatrix2: {labels2}"
        )

    n = matrix1.shape[0]
    # extract upper triangle (excluding diagonal) as flat vectors
    triu_idx = np.triu_indices(n, k=1)
    vec1 = matrix1.to_numpy()[triu_idx]
    vec2 = matrix2.to_numpy()[triu_idx]

    if method == "spearman":
        observed_stat, _ = spearmanr(vec1, vec2)
    elif method == "pearson":
        observed_stat = np.corrcoef(vec1, vec2)[0, 1]
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")

    # permutation test: shuffle the labels of matrix2, recompute correlation
    perm_stats = np.empty(n_permutations)
    idx_range = np.arange(n)
    for p in range(n_permutations):
        perm_order = rng.permutation(idx_range)
        permuted_matrix2 = matrix2.to_numpy()[np.ix_(perm_order, perm_order)]
        perm_vec2 = permuted_matrix2[triu_idx]

        if method == "spearman":
            perm_stat, _ = spearmanr(vec1, perm_vec2)
        else:
            perm_stat = np.corrcoef(vec1, perm_vec2)[0, 1]
        perm_stats[p] = perm_stat

    # two-tailed p-value: how extreme is the observed stat vs the null distribution
    p_value = np.sum(np.abs(perm_stats) >= np.abs(observed_stat)) / n_permutations

    return {
        "statistic": observed_stat,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "method": method,}

# loading csv files 
# reading mmd csv as a data frame 
df_mmd_v2 = pd.read_csv("mmd_matrix.csv", index_col=0)
# reading mtdna csv as a data frame 
df_mtdna = pd.read_csv("mtDNA.csv", index_col=0)
# reading nuclear csv as a dataframe 
df_fst = pd.read_csv("fst_matrix.csv", index_col=0)

# test for MMD matrix vs mtDNA matrix
common_order_mt = df_mtdna.index.tolist()
df_mmd = df_mmd.loc[common_order, common_order]

result_mtDNA = mantel_test(df_mmd, df_mtdna, method="spearman", n_permutations=9999, random_state=42)
print(f"Spearman Mantel statistic for mtDNA matrix: {result_mtDNAtistic']:.4f}")
print(f"P-value for mtDNA ({result_mtDNA['n_permutations']} permutations): {result_mtNDA['p_value']:.4f}")

# test for MMD matrix versus nucelar DNA matrix 
common_order_fst = df_fst.index.tolist()

result_fst = mantel_test(df_mmd, df_fst, method="spearman", n_permutations=9999, random_state=42)
print(f"Spearman Mantel statistic: {result_fst['statistic']:.4f}")
print(f"P-value for fst ({result_fst['n_permutations']} permutations): {result_fst['p_value']:.4f}")

