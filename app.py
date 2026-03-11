import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("🎵 Spotify Song Clustering App")

st.write("This app clusters Spotify songs based on audio features using KMeans.")

# Load dataset
df = pd.read_csv("SpotifyFeatures.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select features
features = df[['danceability','energy','tempo','loudness','valence']]

# Handle missing values
features = features.dropna()

# Normalize data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Cluster selection
k = st.slider("Select Number of Clusters", 2, 10, 4)

# Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

df = df.loc[features.index]
df['cluster'] = clusters

st.subheader("Clustered Data")
st.dataframe(df[['danceability','energy','tempo','loudness','valence','cluster']].head())

# PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_features)

fig, ax = plt.subplots()
scatter = ax.scatter(pca_data[:,0], pca_data[:,1], c=clusters)
ax.set_title("Spotify Song Clusters")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

st.pyplot(fig)

# Cluster Insights
st.subheader("Cluster Insights")
cluster_mean = df.groupby('cluster')[['danceability','energy','tempo','loudness','valence']].mean()
st.dataframe(cluster_mean)
