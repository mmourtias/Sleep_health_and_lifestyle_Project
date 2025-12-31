import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


warnings.filterwarnings("ignore")

# 1) Load data
df = pd.read_csv("data/Sleep_health_cleaned.csv")

features = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    "Systolic",
    "Diastolic",
]

# 2) Ensure numeric + handle missing
X = df[features].apply(pd.to_numeric, errors="coerce")
df = df.loc[X.dropna().index].copy()
X = X.dropna()

# 3) Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Elbow
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertia, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for K-Means")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5) Final KMeans (pick k from elbow)
k_best = 4  # <-- από το elbow
kmeans_final = KMeans(n_clusters=k_best, random_state=42, n_init=10)
df["Cluster_KMeans"] = kmeans_final.fit_predict(X_scaled)

print(f"\nKMeans fitted with k={k_best}")
print("SSE / Inertia:", kmeans_final.inertia_)

print("\n--- Cluster sizes ---")
print(df["Cluster_KMeans"].value_counts().sort_index())

print("\n--- Cluster means ---")
print(df.groupby("Cluster_KMeans")[features].mean().round(3))


# 3. Ιεραρχική Συσταδοποίηση (Hierarchical Clustering)
from sklearn.cluster import AgglomerativeClustering

Z = linkage(X_scaled, method="ward", metric="euclidean")

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    truncate_mode="level", 
    p=5,
    no_labels=True
)
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plt.xlabel("Clusters")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 4. Agglomerative Clustering (choose k from dendrogram)
# --------------------------------------------------
k_hier = 4   # <-- από το dendrogram

agg = AgglomerativeClustering(
    n_clusters=k_hier,
    linkage="ward"
)

df["Cluster_Hierarchical"] = agg.fit_predict(X_scaled)

# --------------------------------------------------
# 5. Cluster profiling (MEANS)
# --------------------------------------------------
print("\n--- Hierarchical cluster sizes ---")
print(df["Cluster_Hierarchical"].value_counts().sort_index())

print("\n--- Hierarchical cluster means ---")
print(
    df.groupby("Cluster_Hierarchical")[features]
    .mean()
    .round(3)
)


# 4. Μέθοδος DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np

print("\nΕκτέλεση DBSCAN...")

# --------------------------------------------------
# 4a) K-distance graph για επιλογή eps
# Κανόνας: n_neighbors = min_samples
# --------------------------------------------------
for min_s in [4, 8, 10, 15]:
    neigh = NearestNeighbors(n_neighbors=min_s)
    neigh.fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)

    k_dist = np.sort(distances[:, -1])

    plt.figure(figsize=(7, 4))
    plt.plot(k_dist)
    plt.title(f"k-distance Graph (k=min_samples={min_s})")
    plt.xlabel("Sorted points")
    plt.ylabel(f"Distance to {min_s}-th nearest neighbor")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 4b) Εφαρμογή DBSCAN

# --------------------------------------------------
eps = 1.2       # <--  από το k-distance
min_samples = 10    

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df["Cluster_DBSCAN"] = dbscan.fit_predict(X_scaled)

# --------------------------------------------------
# 4c) Βασική αναφορά αποτελεσμάτων
# --------------------------------------------------
print("\n--- DBSCAN label counts (including noise=-1) ---")
print(df["Cluster_DBSCAN"].value_counts().sort_index())

n_noise = (df["Cluster_DBSCAN"] == -1).sum()
n_total = df.shape[0]
print(f"\nNoise points: {n_noise}/{n_total} ({n_noise/n_total:.1%})")

n_clusters = df.loc[df["Cluster_DBSCAN"] != -1, "Cluster_DBSCAN"].nunique()
print("Number of clusters (excluding noise):", n_clusters)

# --------------------------------------------------
# 4d) Means ανά cluster (εξαιρούμε noise γιατί δεν είναι cluster)
# --------------------------------------------------
df_core = df[df["Cluster_DBSCAN"] != -1].copy()
if df_core.shape[0] > 0 and df_core["Cluster_DBSCAN"].nunique() > 0:
    print("\n--- DBSCAN cluster means (excluding noise) ---")
    print(df_core.groupby("Cluster_DBSCAN")[features].mean().round(3))
else:
    print("\nNo clusters found (all points labeled as noise). Try adjusting eps/min_samples.")

# --------------------------------------------------
# 4e) Scatterplots (καλύτερο palette για clusters + noise)
# --------------------------------------------------
print("\nΠαραγωγή Scatterplots...")

plot_pairs = [
    ("Age", "Daily Steps"),
    ("Sleep Duration", "Quality of Sleep"),
    ("Stress Level", "Heart Rate"),
    ("Systolic", "Diastolic"),
    ("Physical Activity Level", "Stress Level"),
    ("Sleep Duration", "Stress Level"),
]

for x_feat, y_feat in plot_pairs:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x=x_feat, y=y_feat,
        hue="Cluster_DBSCAN",
        palette="tab10",
        s=60
    )
    plt.title(f"DBSCAN: {x_feat} vs {y_feat} (eps={eps}, min_samples={min_samples})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 4f) Save results
# --------------------------------------------------
out_path = "cleaned_data_clusterAssignmentsDBSCAN.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
