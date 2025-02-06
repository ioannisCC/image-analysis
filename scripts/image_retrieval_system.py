import numpy as np
import pandas as pd
import os

def load_saved_model(hypergraph_file='hypergraph_data.npz', metadata_file='combined_data.csv'):
    if not os.path.exists(hypergraph_file):
        print(f"Saved hypergraph model '{hypergraph_file}' not found. Please run the model-building script first.")
        exit(1)
    data = np.load(hypergraph_file)
    W = data['W']
    
    if not os.path.exists(metadata_file):
        print(f"Metadata file '{metadata_file}' not found. Please ensure you have saved the combined metadata.")
        exit(1)
    combined_df = pd.read_csv(metadata_file)
    
    return W, combined_df

def query_image(W, combined_df, query, top_k=5):
    # Determine query index: if query is int, use it directly; otherwise, search for the image_id.
    if isinstance(query, int):
        query_index = query
    else:
        matches = combined_df[combined_df['image_id'] == query].index
        if len(matches) == 0:
            print(f"No image with id '{query}' found.")
            return
        query_index = matches[0]
    
    # Retrieve top_k similar images using the saved affinity matrix W.
    # We exclude the query image itself.
    sorted_indices = np.argsort(W[query_index])[-top_k*2:][::-1]
    top_indices = [idx for idx in sorted_indices if idx != query_index][:top_k]
    
    query_meta = combined_df.iloc[query_index]
    print(f"Top {top_k} similar images for image {query_meta['image_id']}:")
    for rank, idx in enumerate(top_indices):
        score = W[query_index, idx]
        meta = combined_df.iloc[idx]
        print(f"  {rank + 1}. Image ID: {meta['image_id']}, "
              f"Class ID: {meta['class_id']}, "
              f"Species: {meta['species']}, "
              f"Breed ID: {meta['breed_id']}, "
              f"Score: {score:.4f}")

if __name__ == '__main__':
    # Load saved hypergraph model and metadata.
    W, combined_df = load_saved_model()

    # Ask user for a query: can be an image id or an index.
    user_input = input("Enter image id or index to query: ")
    try:
        query = int(user_input)
    except ValueError:
        query = user_input.strip()
    
    query_image(W, combined_df, query, top_k=5)