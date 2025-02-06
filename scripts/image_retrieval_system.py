import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import os
import pickle

class ImageRetrievalSystem:
    def __init__(self, k=5, L=3):
        self.k = k
        self.L = L
        self.H = None
        self.W = None
        self.image_info_df = None
        self.features_array = None
        
    def save_state(self, filepath):
        """Save the computed structures to disk"""
        state = {
            'H': self.H,
            'W': self.W,
            'image_info_df': self.image_info_df,
            'features_array': self.features_array,
            'k': self.k,
            'L': self.L
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod
    def load_state(cls, filepath):
        """Load a previously saved state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        retrieval_system = cls(k=state['k'], L=state['L'])
        retrieval_system.H = state['H']
        retrieval_system.W = state['W']
        retrieval_system.image_info_df = state['image_info_df']
        retrieval_system.features_array = state['features_array']
        return retrieval_system

    def build_system(self, features_array, image_info_df):
        """Build the complete retrieval system"""
        self.features_array = features_array
        self.image_info_df = image_info_df
        
        # Compute initial distances and similarities
        distance_matrix = cdist(features_array, features_array, metric='euclidean')
        similarity_matrix = np.exp(-distance_matrix)
        
        # Compute initial rankings
        initial_ranks = np.argsort(similarity_matrix, axis=1)[:, ::-1]
        
        # Build hypergraph
        self.H = self._build_hypergraph(initial_ranks)
        
        # Compute final similarity matrix
        self.W = self._compute_final_similarity()
        
    def _build_hypergraph(self, initial_ranks):
        """Build hypergraph with continuous weights"""
        n = len(self.features_array)
        H = np.zeros((n, n))
        
        for i in range(n):
            N_i = initial_ranks[i, :self.k]
            for j in range(n):
                if i == j:
                    continue
                r_sum = 0
                for x in N_i:
                    N_x = initial_ranks[x, :self.k]
                    if j in N_x:
                        try:
                            wp_ix = self._calculate_wp(i, x, initial_ranks)
                            wp_xj = self._calculate_wp(x, j, initial_ranks)
                            r_sum += wp_ix * wp_xj
                        except Exception as e:
                            print(f"Error at i={i}, j={j}, x={x}: {e}")
                H[i, j] = r_sum
        
        return H
    
    def _calculate_wp(self, i, x, initial_ranks):
        """Calculate weight using exponential decay"""
        tau_ix = float(np.where(initial_ranks[i] == x)[0][0] + 1)
        return np.exp(-tau_ix / self.k)
    
    def _compute_final_similarity(self):
        """Compute final similarity matrix W"""
        # Convert to sparse for efficiency
        H_sparse = sp.csr_matrix(self.H)
        
        # Compute similarity matrices
        Sh = H_sparse @ H_sparse.T
        Sv = H_sparse.T @ H_sparse
        S = Sh.multiply(Sv).toarray()
        
        # Compute Cartesian product
        w = np.sum(self.H, axis=1)
        C = (H_sparse.T.multiply(w)) @ H_sparse
        
        # Final similarity matrix
        return S * C.toarray()
    
    def query_similar_images(self, query_idx, top_k=5):
        """Find top-k similar images for a given query image index"""
        if self.W is None:
            raise ValueError("System not initialized. Call build_system first.")
            
        # Get top indices excluding the query image itself
        top_indices = [idx for idx in np.argsort(self.W[query_idx])[-top_k*2:][::-1] 
                      if idx != query_idx][:top_k]
        
        # Get query image metadata
        query_metadata = self.image_info_df.iloc[query_idx]
        results = []
        
        # Format results
        for rank, idx in enumerate(top_indices):
            score = self.W[query_idx, idx]
            similar_image = self.image_info_df.iloc[idx]
            results.append({
                'rank': rank + 1,
                'image_id': similar_image['image_id'],
                'class_id': similar_image['class_id'],
                'species': similar_image['species'],
                'breed_id': similar_image['breed_id'],
                'similarity_score': score
            })
            
        return {
            'query_image': {
                'image_id': query_metadata['image_id'],
                'class_id': query_metadata['class_id'],
                'species': query_metadata['species'],
                'breed_id': query_metadata['breed_id']
            },
            'similar_images': results
        }

# Example usage:
if __name__ == "__main__":
    # Initialize and build system
    retrieval_system = ImageRetrievalSystem(k=5, L=3)
    
    # If starting fresh:
    features_array = np.load('/Users/ioannis/Library/CloudStorage/OneDrive-unipi.gr/Desktop/ioannis/university/7th semester/image analysis/assignment/scripts/features.npy')
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
    retrieval_system.build_system(features_array, image_info_df)
    
    # Save the computed structures
    retrieval_system.save_state('retrieval_system_state.pkl')
    
    # Later, to load and query:
    loaded_system = ImageRetrievalSystem.load_state('retrieval_system_state.pkl')
    results = loaded_system.query_similar_images(query_idx=371, top_k=5)
    
    # Print results
    print(f"Query Image: {results['query_image']['image_id']}")
    print("\nTop 5 Similar Images:")
    for img in results['similar_images']:
        print(f"{img['rank']}. {img['image_id']} (Score: {img['similarity_score']:.4f})")