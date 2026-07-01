# imports
import pandas as pd
import os
import umap
from sklearn.preprocessing import StandardScaler

# setting up files directories
base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
input_file = os.path.join(base_dir, "features", "perch_embeddings.csv")
output_dir = os.path.join(base_dir, "results")

os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "umap_coordinates.csv")

# reading features csv file
features_df = pd.read_csv(input_file)
# looking only at vectors 
features = features_df.filter(like="vector")
# scaling data
features_scaled = StandardScaler().fit_transform(features)

# initialising umap and applying to scaled features 
fit = umap.UMAP(n_neighbors = max(5, int(0.001 * features.shape[0])),  # setting n_neighbours as dynamic to scale with dataset
                n_components= 2, 
                random_state = 42,
                metric = 'cosine') 
reduced_emmbeddings = fit.fit_transform(features_scaled)

# creating data frame with coordinates 
umap_df = pd.DataFrame(reduced_emmbeddings, columns=["UMAP_1", "UMAP_2"])
umap_df.insert(0, "Filename", features_df["Filename"])
print(umap_df.head())

# saving dataframe as csv 
umap_df.to_csv(output_file, index=False)