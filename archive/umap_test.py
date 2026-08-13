import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

# reading features csv file
features_df = pd.read_csv("/Users/anyasen/Documents/project/embeddings/audio_features.csv")

# looking only at vectors 
features = features_df.filter(like="vector")
# scaling data
features_scaled = StandardScaler().fit_transform(features)

# initialising umap and applying to scaled features 
fit = umap.UMAP(n_neighbors = max(5, int(0.001 * features.shape[0])), n_components= 2, random_state = 42) # setting n_neighbours as dynamic to scale with dataset
reduced_emmbeddings = fit.fit_transform(features_scaled)

# creating data frame with coordinates 
umap_df = pd.DataFrame(reduced_emmbeddings, columns=["UMAP_1", "UMAP_2"])
umap_df.insert(0, "Filename", features_df["Filename"])

print(umap_df.head())