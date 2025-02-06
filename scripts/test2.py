import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import os


# path to 'list.txt' file
dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "annotations", "list.txt")

image_info = []

with open(dataset_path, 'r') as file:
    for line in file:
        # skip the header lines
        if line.startswith("#"):
            continue
        
        # split the line into parts (separated by spaces)
        parts = line.strip().split()
        
        # ensure that lines are consistent
        if len(parts) == 4:
            image_id = parts[0]  # image id (e.g., Abyssinian_100)
            class_id = int(parts[1])  # class id (1: Cat, 2: Dog)
            species = 1 if image_id[0].isupper() else 2  # 1 for cat, 2 for dog
            breed_id = int(parts[3])  # breed id
            
            image_info.append({
                'image_id': image_id,
                'class_id': class_id,
                'species': species,
                'breed_id': breed_id
            })

# convert the list of dictionaries into a DataFrame
image_info_df = pd.DataFrame(image_info)

print(image_info_df.head())

# load features
features_array = np.load('/Users/ioannis/Library/CloudStorage/OneDrive-unipi.gr/Desktop/ioannis/university/7th semester/image analysis/assignment/scripts/features.npy')
print(features_array.shape)

# --- Step 1: Compute distances and similarities ---
# calculate the distance of each images with all the training images using euclidean distance
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

def calculate_distances(features, features_array):
    return cdist(features, features_array, metric='euclidean')

# Compute the distance matrix (assuming calculate_distances is defined)
distance_matrix = calculate_distances(features_array, features_array)
print(f'Distance Matrix (first 10 rows):\n{distance_matrix[:10]}\n')

# Use an exponential kernel instead of 1/(1+δ)
similarity_matrix = np.exp(-distance_matrix)
print(f'Similarity Matrix (first 10 rows):\n{similarity_matrix[:10]}\n')

# Compute initial ranking order (indices in descending order of similarity)
initial_ranks = np.argsort(similarity_matrix, axis=1)[:, ::-1]
print(f'Initial Ranks (first 10 rows):\n{initial_ranks[:10]}\n')

initial_ranks_values = np.sort(similarity_matrix, axis=1)[:, ::-1]
print(f'Initial Sorted Similarity Values (first 10 rows):\n{initial_ranks_values[:10]}\n')

# --- Step 2: Rank Normalization ---
# This is a simple normalization. You can change the method as needed.
def rank_normalization(ranks, L):
    # For each row, assign a normalized weight of 1 for the top-L positions and 0 for others.
    norm_ranks = np.zeros_like(ranks, dtype=float)
    norm_ranks[:, :L] = 1.0
    return norm_ranks

L = 3
normalized_ranks = rank_normalization(initial_ranks, L)
print(f'Normalized Ranks (first 10 rows):\n{normalized_ranks[:10]}\n')

# --- Step 3: Build Hypergraph with Continuous Weights ---

# Modified weighting function: using an exponential decay on the rank position.
def calculate_wp(i, x, initial_ranks, k):
    """
    Calculate weight wp(i,x) using exponential decay.
    i: query image index
    x: neighbor index
    """
    # Add 1 to the rank position (avoid log(0) issues) and use exponential decay.
    tau_ix = float(np.where(initial_ranks[i] == x)[0][0] + 1)
    return np.exp(-tau_ix / k)

def build_hypergraph(features_array, initial_ranks, k):
    """
    Build hypergraph incidence matrix H using continuous weights.
    """
    n = len(features_array)
    H = np.zeros((n, n))
    for i in range(n):
        # Get the top-k neighbors for image i
        N_i = initial_ranks[i, :k]
        for j in range(n):
            if i == j:
                continue
            r_sum = 0
            # For each neighbor x in N(i,k)
            for x in N_i:
                N_x = initial_ranks[x, :k]
                if j in N_x:
                    try:
                        wp_ix = calculate_wp(i, x, initial_ranks, k)
                        wp_xj = calculate_wp(x, j, initial_ranks, k)
                        r_sum += wp_ix * wp_xj
                    except Exception as e:
                        print(f"Error at i={i}, j={j}, x={x}: {e}")
            H[i, j] = r_sum
            if r_sum > 0:
                print(f"Non-zero weight: H[{i},{j}] = {r_sum}")
    return H

k = 5  # You can try different values here.
H = build_hypergraph(features_array, initial_ranks, k)

print(f"\nMatrix H statistics:")
print(f"Shape: {H.shape}")
print(f"Non-zero elements: {np.count_nonzero(H)}")
print(f"Max value: {np.max(H)}")
print(f"First 5 rows of H:\n{H[:5, :5]}\n")

# --- Step 4: Compute Hypergraph-Based Similarity Matrices ---

# Hyperedge-based similarity matrix (Sh)
Sh = np.dot(H, H.T)
print("Sh (Hyperedge-based Similarity Matrix):")
print(Sh)

# Vertex-based similarity matrix (Sv)
Sv = np.dot(H.T, H)
print("Sv (Vertex-based Similarity Matrix):")
print(Sv)

# Combine the two similarities using Hadamard (elementwise) product
S = np.multiply(Sh, Sv)
print("Combined Similarity Matrix S (Hadamard product):")
print(S)

# --- Step 5: Use Sparse Matrix Operations for Cartesian Product ---

# Compute hyperedge weights by summing each row of H
w = np.sum(H, axis=1)
H_sparse = sp.csr_matrix(H)
# Compute Cartesian product matrix C
C = (H_sparse.T.multiply(w)) @ H_sparse
# Combine with S to form final similarity matrix W
W = S * C.toarray()
print("Final Similarity Matrix W:")
print(W)

# --- Step 6: Optional Iterative Ranking Refinement ---

# Run a few iterations to refine rankings based on W.
iterations = 2
T = initial_ranks.copy()
for t in range(iterations):
    T = np.argsort(W, axis=1)[:, ::-1]
    print(f"Iteration {t+1} updated rankings (first 10 rows):")
    print(T[:10])

# --- Step 7: Retrieve and Display Top-k Similar Images ---

# Combine features and metadata into a DataFrame (if not already done)
combined_data = []
for i in range(len(features_array)):
    row = image_info_df.iloc[i]
    combined_data.append({
        'image_id': row['image_id'],
        'class_id': row['class_id'],
        'species': row['species'],
        'breed_id': row['breed_id'],
        'features': features_array[i],
        'initial_ranks_subset': initial_ranks[i],
        'initial_sorted_values_subset': initial_ranks_values[i],
        'normalized_ranks': normalized_ranks[i]
    })
combined_df = pd.DataFrame(combined_data)
print("Combined DataFrame head:")
print(combined_df.head())

top_k = 5
for i in range(W.shape[0]):
    # Exclude the image itself and get top similar indices
    top_indices = [idx for idx in np.argsort(W[i])[-top_k*2:][::-1] if idx != i][:top_k]
    query_image_metadata = combined_df.iloc[i]
    print(f"Top {top_k} similar images for image {query_image_metadata['image_id']}:")
    print(f"  Class ID: {query_image_metadata['class_id']}, Species: {query_image_metadata['species']}, Breed ID: {query_image_metadata['breed_id']}\n")
    for rank, idx in enumerate(top_indices):
        score = W[i, idx]
        similar_image_metadata = combined_df.iloc[idx]
        print(f"  {rank + 1}. Image ID: {similar_image_metadata['image_id']}, "
              f"Class ID: {similar_image_metadata['class_id']}, "
              f"Species: {similar_image_metadata['species']}, "
              f"Breed ID: {similar_image_metadata['breed_id']}, "
              f"Score: {score:.4f}")
    print("=" * 50)