# imports 
import pandas as pd 
import numpy as np
import math
from itertools import combinations
import os

# list of subpopulations  
subpops = [
    "SRKW", "NRKW", "AR_AD", "AR_AB", "OKW", "WCT", "GAT", "AT1"]

# constructing dictionary of pairwise distances (evolutionary distances)- manually matched using coordinates 
pairwise_distances = { ("SRKW", "NRKW"): 1.00148,
                        ("AR_AB", "SRKW"): 1.00148,
                        ("AR_AD", "SRKW"): 0,
                        ("AR_AB", "NRKW"): 0,
                        ("AR_AD", "NRKW"): 1.00148,
                        ("AR_AB", "AR_AD"): 1.00148,
                        ("OKW", "SRKW"): 1.01808,
                        ("OKW", "NRKW"): 1.00148 + 1.01808, 
                        ("OKW", "AR_AB"): 1.00148 + 1.01808,
                        ("OKW", "AR_AD"): 1.01808,
                        ("WCT", "SRKW"): 1.00525 + 1.00364 + 2.52708 + 2.52708 + 1.00794,
                        ("WCT", "NRKW"): 1.00525 + 1.00364 + 2.52708 + 2.52708 + 1.00794 + 1.00148,
                        ("WCT", "AR_AB"): 1.00525 + 1.00364 + 2.52708 + 2.52708 + 1.00794 + 1.00148,
                        ("WCT", "AR_AD"): 1.00525 + 1.00364 + 2.52708 + 2.52708 + 1.00794,
                        ("WCT", "OKW"): 1.00525 + 1.00364 + 2.52708 + 2.52708 + 1.00794 + 1.01808,
                        ("GAT_A", "SRKW"): 1.00364 + 2.52708 + 2.52708 + 1.00794,
                        ("GAT_A", "NRKW"): 1.00364 + 2.52708 + 2.52708 + 1.00794 + 1.00148,
                        ("GAT_A", "AR_AB"): 1.00364 + 2.52708 + 2.52708 + 1.00794 + 1.00148,
                        ("GAT_A", "AR_AD"): 1.00364 + 2.52708 + 2.52708 + 1.00794,
                        ("GAT_A", "OKW"): 1.00364 + 2.52708 + 2.52708 + 1.00794 + 1.01808,
                        ("GAT_A", "WCT"): 1.00525,
                        ("GAT_A", "GAT_B"): 1.00364,
                        ("GAT_B", "SRKW"): 2.52708 + 2.52708 + 1.00794, 
                        ("GAT_B", "NRKW"): 2.52708 + 2.52708 + 1.00794 + 1.00148,
                        ("GAT_B", "AR_AB"): 2.52708 + 2.52708 + 1.00794 + 1.00148,
                        ("GAT_B", "AR_AD"): 2.52708 + 2.52708 + 1.00794,
                        ("GAT_B", "OKW"): 2.52708 + 2.52708 + 1.00794 + 1.01808,
                        ("GAT_B", "WCT"): 1.00364 + 1.00525,
                        ("AT1", "SRKW"): 2.01176 + 1.00364 + 2.52708 + 2.52708 + 1.00794,
                        ("AT1", "NRKW"): 2.01176 + 1.00364 + 2.52708 + 2.52708 + 1.00794 + 1.00148,
                        ("AT1", "AR_AB"):2.01176 + 1.00364 + 2.52708 + 2.52708 + 1.00794 + 1.00148,
                        ("AT1", "AR_AD"):2.01176 + 1.00364 + 2.52708 + 2.52708 + 1.00794,
                        ("AT1", "OKW"): 2.01176 + 1.00364 + 2.52708 + 2.52708 + 1.00794 + 1.01808,
                        ("AT1", "WCT"): 2.01176 + 1.00525,
                        ("AT1", "GAT_A"): 2.01176,
                        ("AT1", "GAT_B"): 2.01176 + 1.00364}

# building pariwise matrix 
raw_leaves = sorted({s for pair in pairwise_distances for s in pair})
n_raw = len(raw_leaves)
raw_idx = {s: i for i, s in enumerate(raw_leaves)}

# pre-filling with nan 
raw_matrix = np.full((n_raw, n_raw), np.nan)
# filling diagonal with zeroes
np.fill_diagonal(raw_matrix, 0.0)

# filling matrix symmetrically
for (a, b), dist in pairwise_distances.items():
    i, j = raw_idx[a], raw_idx[b]
    raw_matrix[i, j] = dist
    raw_matrix[j, i] = dist

# enusuring every pair is present berfore merging     
missing_raw = [
    (raw_leaves[i], raw_leaves[j])
    for i, j in combinations(range(n_raw), 2)
    if np.isnan(raw_matrix[i, j])
]
if missing_raw:
    raise ValueError(f"Missing {len(missing_raw)} raw pairs before merging: {missing_raw}")
print(f"Raw matrix complete: {n_raw} leaves, all {n_raw*(n_raw-1)//2} pairs present.")

df_raw = pd.DataFrame(raw_matrix, index=raw_leaves, columns=raw_leaves)

# merging vocal clans for subpops that were split 
merge_map = {
    # merging two SAR clans 
    "SAR": ["AR_AD", "AR_AB"],
    # merging two GAT groups 
    "GAT": ["GAT_A", "GAT_B"],}

# defining function to collapse groups into single group be averaging distance to every other label 
def merge_groups(df, merge_map):
    
    df = df.copy()
    all_grouped = {leaf for group in merge_map.values() for leaf in group}

    # replacing groups with single merged label 
    final_labels = []
    seen_merged = set()
    for label in df.index:
        if label in all_grouped:
            merged_name = next(name for name, members in merge_map.items() if label in members)
            if merged_name not in seen_merged:
                final_labels.append(merged_name)
                seen_merged.add(merged_name)
        else:
            final_labels.append(label)

    n = len(final_labels)
    merged_matrix = np.full((n, n), np.nan)

    # defining function to return orihinal leaf level label for groups that were merged 
    def members_of(label):
        return merge_map.get(label, [label])
    
    # averaging pairs of distances with other subpops for each member in group 
    for i, li in enumerate(final_labels):
        for j, lj in enumerate(final_labels):
            if i == j:
                merged_matrix[i, j] = 0.0
                continue
            members_i = members_of(li)
            members_j = members_of(lj)
            dists = [df.loc[mi, mj] for mi in members_i for mj in members_j]
            merged_matrix[i, j] = np.mean(dists)

    return pd.DataFrame(merged_matrix, index=final_labels, columns=final_labels)

df_merged = merge_groups(df_raw, merge_map)

# reordering to match acoustic and nuclear distance matrices 
subpops = ["AT1","GAT", "NRKW", "OKW", "SAR", "SRKW", "WCT"]
df_mtdna = df_merged.loc[subpops, subpops]
print(df_mtdna)

# saving dataframe as csv file 
df_mtdna.to_csv(output_mtdna)

