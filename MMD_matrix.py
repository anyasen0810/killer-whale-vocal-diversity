import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# defining function that caculates MMD between two distributions 
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