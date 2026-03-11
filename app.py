import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("Spotify Song Clustering App")

st.write("Upload Spotify Dataset CSV file")

# Upload dataset
uploaded_file = st.file_uploader("Upload SpotifyFeatures.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select features
    features = df[['danceability','energy','tempo','loudness','valence']]

    # Remove missing values
    features = features.dropna()

    # Normalize data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Select number of clusters
    k = st.slider("Select number of clusters", 2, 10, 4)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    df = df.loc[features.index]
    df['cluster'] = clusters

    st.subheader("Clustered Data")
    st.write(df.head())

    # PCA Visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_features)

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_data[:,0], pca_data[:,1], c=clusters)
    ax.set_title("Spotify Song Clusters")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    st.pyplot(fig)

    # Cluster insights
    st.subheader("Cluster Insights")
    st.write(df.groupby('cluster')[['danceability','energy','tempo','loudness','valence']].mean())

else:
    st.warning("Please upload the SpotifyFeatures.csv file to continue.")
