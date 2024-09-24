import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the dataset (dummy path for now)
df = pd.read_csv('complete_dataset.csv')

# Simulate a small dataset for demonstration
data = {
    'Insulin Levels': np.random.rand(100),
    'Age': np.random.randint(20, 70, 100),
    'BMI': np.random.rand(100) * 30 + 15,
    'Blood Pressure': np.random.rand(100) * 40 + 80,
    'Cholesterol Levels': np.random.rand(100) * 100 + 150,
    'Waist Circumference': np.random.rand(100) * 50 + 50,
    'Blood Glucose Levels': np.random.rand(100) * 100 + 50,
}
df = pd.DataFrame(data)

# Select relevant features for clustering
features = [
    'Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 
    'Cholesterol Levels', 'Waist Circumference', 'Blood Glucose Levels'
]
X = df[features]

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Standardize the data using StandardScaler from sklearn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the K-Means clustering function using TensorFlow 2.4+ compatible code
def kmeans(X, k, num_iters=100):
    # Randomly initialize centroids
    initial_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = tf.Variable(X[initial_indices], dtype=tf.float32)

    for _ in range(num_iters):
        # Calculate distances to centroids
        distances = tf.norm(tf.expand_dims(X, axis=1) - centroids, axis=2)
        
        # Assign clusters based on closest centroid
        clusters = tf.argmin(distances, axis=1)

        # Update centroids
        for i in range(k):
            points_in_cluster = tf.gather(X, tf.where(clusters == i))
            if tf.shape(points_in_cluster)[0] > 0:
                centroids[i].assign(tf.reduce_mean(points_in_cluster, axis=0))

    return clusters, centroids

# Determine the optimal number of clusters using the Elbow method
inertia = []
K = range(1, 11)

for k in K:
    clusters, centroids = kmeans(X_scaled, k)
    distances = tf.norm(tf.expand_dims(X_scaled, axis=1) - centroids, axis=2)
    inertia.append(tf.reduce_sum(tf.reduce_min(distances, axis=1)).numpy())

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# Choose the optimal number of clusters (e.g., k=3 from the elbow method)
optimal_k = 3
clusters, centroids = kmeans(X_scaled, optimal_k)
df['Cluster'] = clusters.numpy()

# Plot the clustered data
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Blood Glucose Levels', y='BMI', hue='Cluster', palette='viridis', s=100)
plt.title('Clustered Data Visualization')
plt.xlabel('Blood Glucose Levels')
plt.ylabel('BMI')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Analyze clusters
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    print(df[df['Cluster'] == cluster][features].describe())