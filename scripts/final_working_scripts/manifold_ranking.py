import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import os
from sklearn.model_selection import train_test_split

# path to 'list.txt' file containg all the metadata
list_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "annotations", "list.txt")

image_info = []

with open(list_file, 'r') as file:
    for line in file:
        # skip the header lines
        if line.startswith("#"):
            continue
        
        # split the line into parts (separated by spaces)
        parts = line.strip().split()
        
        # each entry consists of 4 parts
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
features_array = np.load('/Users/ioannis/Library/CloudStorage/OneDrive-unipi.gr/Desktop/ioannis/university/7th semester/image analysis/assignment/artifacts/features.npy')
print(features_array.shape)

# --- step 1: compute distances and similarities ---

# calculate the distance of each images with all the training images using cosine distance
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
def calculate_distances(features, features_array):
    return cdist(features, features_array, metric='cosine')

# compute the distance matrix
distance_matrix = calculate_distances(features_array, features_array)
print(f'Distance Matrix (first 10 rows):\n{distance_matrix[:10]}\n')

# exponential kernel for the similarity matrix
similarity_matrix = np.exp(-distance_matrix)
print(f'Similarity Matrix (first 10 rows):\n{similarity_matrix[:10]}\n')

# initial ranking order (indices in descending order of similarity)
initial_ranks = np.argsort(similarity_matrix, axis=1)[:, ::-1]
print(f'Initial Ranks (first 10 rows):\n{initial_ranks[:10]}\n')

# initial ranking order values (values in descending order of similarity)
initial_ranks_values = np.sort(similarity_matrix, axis=1)[:, ::-1]
print(f'Initial Sorted Similarity Values (first 10 rows):\n{initial_ranks_values[:10]}\n')

# --- step 2: rank normalization ---

def rank_normalization(ranks, L): # L are top positions to consider
    num_items = ranks.shape[0]
    
    # reciprocal rank positions
    reciprocal_ranks = 1.0 / (ranks + 1)
    
    # compute the new similarity measure ⇢n
    rho_n = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            rho_n[i, j] = reciprocal_ranks[i, j] + reciprocal_ranks[j, i]
            # rho_n[i, j] = 2 * L - (ranks[i, j] + ranks[j, i])
    
    # update the top-L positions based on the new similarity measure
    updated_ranks = np.argsort(-rho_n, axis=1)[:, :L]
    
    return updated_ranks

L = 5
normalized_ranks = rank_normalization(initial_ranks, L)
print(f'Normalized Ranks (first 10 rows):\n{normalized_ranks[:10]}\n')

# --- step 3: build hypergraph with weights ---

# logarithmic decay to assign weights
def calculate_wp(i, x, initial_ranks, k):
    # i: item of which the weight is calculated
    # x: item of which weight we want to calculate 
    # k: top items to retrieve (controls rate of decay)
    tau_ix = np.where(initial_ranks[i] == x)[0][0] + 1  # Position of x in τi
    return 1 - np.log(tau_ix) / np.log(k)

# represent the incidence matrix 
def build_hypergraph(features_array, initial_ranks, k):
    n = len(features_array)
    H = np.zeros((n, n))
    for i in range(n):
        # get the top-k neighbors for image i
        N_i = initial_ranks[i, :k]
        for j in range(n):
            if i == j:
                continue
            r_sum = 0
            # for each neighbor x in N(i,k)
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

k = 5 # top neighbors to consider
H = build_hypergraph(features_array, initial_ranks, k)

print(f"\nMatrix H statistics:")
print(f"Shape: {H.shape}")
print(f"Non-zero elements: {np.count_nonzero(H)}")
print(f"Max value: {np.max(H)}")
print(f"First 5 rows of H:\n{H[:5, :5]}\n")

# --- step 4: hypergraph-based similarity matrices ---

# hyperedge-based similarity matrix (Sh)
Sh = np.dot(H, H.T)
print("Sh (Hyperedge-based Similarity Matrix):")
print(Sh)

# vertex-based similarity matrix (Sv)
Sv = np.dot(H.T, H)
print("Sv (Vertex-based Similarity Matrix):")
print(Sv)

# combine the two similarities using Hadamard (elementwise) product
S = np.multiply(Sh, Sv)
print("Combined Similarity Matrix S (Hadamard product):")
print(S)

# --- step 5: cartesian product ---

# compute hyperedge weights by summing each row of H
w = np.sum(H, axis=1)
H_sparse = sp.csr_matrix(H)

# use sparse matrix operations since H is sparse

#  Cartesian product matrix C
C = (H_sparse.T.multiply(w)) @ H_sparse
# combine with S to form final affinity matrix W
W = S * C.toarray()
print("Final Affinity Matrix W:")
print(W)

# --- step 6: iterative ranking ---
# version showed in class decreases precision by almost 50%
# refine rankings based on W 
# iterations = 2
# W_t = W.copy()  # initial W(0)
# T = initial_ranks.copy()  # initial T(0)

# for t in range(iterations):
#     # update T(t) based on current W(t)
#     T = np.argsort(W_t, axis=1)[:, ::-1]
#     print(f"Iteration {t+1} updated rankings (first 10 rows):")
#     print(T[:10])
    
#     # reconstruct hypergraph using new rankings T to get W(t+1)
#     # build new hypergraph with new rankings
#     H = build_hypergraph(features_array, T, k)
    
#     # compute new Sh and Sv
#     Sh = np.dot(H, H.T)
#     Sv = np.dot(H.T, H)
#     S = np.multiply(Sh, Sv)
    
#     # compute new Cartesian product
#     w = np.sum(H, axis=1)
#     H_sparse = sp.csr_matrix(H)
#     C = (H_sparse.T.multiply(w)) @ H_sparse
    
#     # update W(t+1)
#     W_t = S * C.toarray()

# # final W_t back to W
# W = W_t.copy()

# iterative ranking (alternative version) decreases precision by almost 1%

# def iterative_ranking_simple(W_initial, iterations=2):
#     """
#     Simpler iterative ranking that only updates rankings based on W
#     without reconstructing the hypergraph.
#     """
#     W = W_initial.copy()
    
#     for t in range(iterations):
#         # Update rankings based on current W
#         T = np.argsort(W, axis=1)[:, ::-1]
#         print(f"Iteration {t+1} updated rankings (first 10 rows):")
#         print(T[:10])
        
#         # Update W using current rankings
#         # Just update the similarity values based on ranking positions
#         W_new = np.zeros_like(W)
#         n = W.shape[0]
#         for i in range(n):
#             for j in range(n):
#                 # Use ranking position to determine new similarity
#                 pos_i = np.where(T[i] == j)[0][0]
#                 pos_j = np.where(T[j] == i)[0][0]
#                 W_new[i,j] = 1.0 / (pos_i + pos_j + 1)  # +1 to avoid division by zero
        
#         W = W_new

#     return W

# W = iterative_ranking_simple(W)

# --- step 7: save for later use ---

# combine features and metadata into a DataFrame
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

# get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
artifacts_path = os.path.abspath(os.path.join(os.path.dirname(current_script_path), '..', '..', 'artifacts'))
os.makedirs(artifacts_path, exist_ok=True) 

# save hypergraph data
hypergraph_file = os.path.join(artifacts_path, 'hypergraph_data.npz')
np.savez(hypergraph_file, H=H, Sh=Sh, Sv=Sv, W=W) # compressed numpy file
print("Hypergraph data saved to 'hypergraph_data.npz'.")

# save metadata (combined DataFrame)
combined_data_file = os.path.join(artifacts_path, 'combined_data.csv')
combined_df.to_csv(combined_data_file, index=False) # csv file
print("Metadata saved to 'combined_data.csv'.")

# --- step 8: print results ---

# evaluate precision with ground truth
def evaluate_precision(W, combined_df, query_index, k=5):
    # get ground truth: images of same breed/class as query
    query_meta = combined_df.iloc[query_index]
    ground_truth = combined_df[
        (combined_df['breed_id'] == query_meta['breed_id']) & 
        (combined_df.index != query_index)
    ].index.tolist()
    
    # get top k retrieved images
    sorted_indices = np.argsort(W[query_index])[-k*2:][::-1]
    retrieved = [idx for idx in sorted_indices if idx != query_index][:k]
    
    # calculate precision
    relevant = sum(1 for idx in retrieved if idx in ground_truth)
    precision = relevant / k
    
    return precision

# print all the results
top_k = 5
for i in range(W.shape[0]):
    # exclude the image itself and get top similar indices
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


# query a specific image
query_image_index = 371  # by index
# or
query_image_id = "Abyssinian_100"  # by id

# retrieve the row corresponding to the query image
if query_image_index is not None:
    query_image_metadata = combined_df.iloc[query_image_index]
elif query_image_id is not None:
    query_image_metadata = combined_df[combined_df['image_id'] == query_image_id].iloc[0]
else:
    raise ValueError("Please provide either query_image_index or query_image_id.")

# get top-k most similar images
top_k = 5
query_image_index = query_image_metadata.name
# exclude itself from the similar images
top_indices = [idx for idx in np.argsort(W[query_image_index])[-top_k*2:][::-1] if idx != query_image_index][:top_k]

print(f"Top {top_k} similar images for query image {query_image_metadata['image_id']}:")
print(f"Precision: " + str(evaluate_precision(W, combined_df, query_image_index, k=5)))
print(f"  Class ID: {query_image_metadata['class_id']}, Species: {query_image_metadata['species']}, Breed ID: {query_image_metadata['breed_id']}\n")
for rank, idx in enumerate(top_indices):
    score = W[query_image_index, idx]
    similar_image_metadata = combined_df.iloc[idx]
    print(f"  {rank + 1}. Image ID: {similar_image_metadata['image_id']}, "
          f"Class ID: {similar_image_metadata['class_id']}, "
          f"Species: {similar_image_metadata['species']}, "
          f"Breed ID: {similar_image_metadata['breed_id']}, "
          f"Score: {score:.4f}")


# --- step 9: evaluation ---

# split the dataset into train and test sets.
train_indices, test_indices = train_test_split(range(len(combined_df)), test_size=0.2, random_state=42)

# build ground truth mapping:
# for each test image, ground_truth[idx] is the set of indices of images with the same class and breed.
ground_truth = {}
for idx in test_indices:
    row = combined_df.iloc[idx]
    class_id = row['class_id']
    breed_id = row['breed_id']
    # Get indices of all images with the same class_id and breed_id.
    ground_truth[idx] = set(combined_df[(combined_df['class_id'] == class_id) & (combined_df['breed_id'] == breed_id)].index.tolist())

# helper function to compute Discounted Cumulative Gain.
def dcg(relevances):
    return sum(rel / np.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))

k_values = [5]  # top-k values for evaluation
results = {}

for k in k_values:
    precision_scores = []
    ap_scores = []
    ndcg_scores = []
    
    for idx in test_indices:
        query_image_index = idx
        # get full ranking for the query image (excluding the query image itself).
        ranking = np.argsort(W[query_image_index])[::-1]
        ranking = [i for i in ranking if i != query_image_index]
        top_k_indices = ranking[:k]
        
        relevant_set = ground_truth[idx]
        num_relevant = len(relevant_set)
        retrieved_relevant = [i for i in top_k_indices if i in relevant_set]
        num_retrieved_relevant = len(retrieved_relevant)
        
        # precision
        precision = num_retrieved_relevant / k        
        # Average Precision (AP) for the top-k list.
        ap = 0.0
        hit_count = 0
        for rank, i in enumerate(top_k_indices, start=1):
            if i in relevant_set:
                hit_count += 1
                ap += hit_count / rank
        if hit_count > 0:
            ap /= hit_count
        
        # NDCG for the top-k list.
        actual_relevances = [1 if i in relevant_set else 0 for i in top_k_indices]
        ideal_relevances = sorted(actual_relevances, reverse=True)
        actual_dcg = dcg(actual_relevances)
        ideal_dcg = dcg(ideal_relevances) if ideal_relevances else 0
        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        precision_scores.append(precision)
        ap_scores.append(ap)
        ndcg_scores.append(ndcg)
    
    results[k] = {
        'Precision': np.mean(precision_scores),
        'MAP': np.mean(ap_scores),
        'NDCG': np.mean(ndcg_scores)
    }

# results.
print("Evaluation Results:")
for k_val, metrics in results.items():
    print(f"Top-{k_val}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()