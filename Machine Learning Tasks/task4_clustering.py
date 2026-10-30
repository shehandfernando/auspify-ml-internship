import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Dataset.csv')

# ==========================================
# Step 1: Prepare features
# ==========================================
# We will cluster content based on genres/listed_in
df['listed_in'] = df['listed_in'].fillna('')

# Convert genres into numerical features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['listed_in'])

# ==========================================
# Step 2: Apply clustering algorithms
# ==========================================
# Group titles into 5 distinct clusters using K-Means
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X)

# Assign the identified cluster back to the dataframe
df['Cluster'] = kmeans.labels_

# ==========================================
# Step 3 & 4: Visualize clusters
# ==========================================
# We use PCA (Principal Component Analysis) to reduce the high-dimensional
# TF-IDF data down to 2 dimensions so we can plot it on an X/Y graph.
pca = PCA(n_components=2, random_state=42)
reduced_features = pca.fit_transform(X.toarray())
df['pca_x'] = reduced_features[:, 0]
df['pca_y'] = reduced_features[:, 1]

# Create the visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca_x', y='pca_y', hue='Cluster', palette='viridis', data=df, alpha=0.6)
plt.title('Netflix Content Segmentation (K-Means Clustering)')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(title='Cluster')
plt.savefig('cluster_visualization.png') # Saves the chart as an image
print("Visualization saved as 'cluster_visualization.png'")

# ==========================================
# Step 5: Interpret cluster characteristics
# ==========================================
print("\nTop genres per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names_out()

for i in range(num_clusters):
    print(f"Cluster {i}: ", end='')
    # Print the top 3 genre terms characterizing this cluster
    top_terms = [terms[ind] for ind in order_centroids[i, :3]]
    print(", ".join(top_terms))