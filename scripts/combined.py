import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import scipy.sparse as sp
from datasets import ImageDataset
import os

class HypergraphImageRetrieval:
    def __init__(self, feature_extractor='resnet18', k=5):
        """Initialize the image retrieval system.
        
        Args:
            feature_extractor: Name of pretrained model to use.
            k: Number of nearest neighbors for hypergraph construction.
        """
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature extractor
        if feature_extractor == 'resnet18':
            model = models.resnet50(pretrained=True)
            # Remove final classification layer
            self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        else:
            raise ValueError("Unsupported feature extractor")
            
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        """Extract deep features from an image."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(image)
        # Flatten the features and convert to numpy
        return features.cpu().squeeze().flatten().numpy()
    
    def build_hypergraph(self, features):
        """Construct hypergraph from feature vectors.
        
        (Existing hypergraph code remains here if needed.)
        """
        N = features.shape[0]
        # Compute pairwise distances (example using Euclidean norm)
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i] = np.linalg.norm(features - features[i], axis=1)
                
        # Build a binary incidence matrix H using k-nearest neighbors
        H = np.zeros((N, N))
        for i in range(N):
            idx = np.argsort(dist_matrix[i])[1:self.k+1]  # skip self (index 0)
            H[i, idx] = 1
            
        # Compute hyperedge weights (if needed)
        W = np.zeros(N)
        for i in range(N):
            neighbors = np.where(H[i] > 0)[0]
            for j in neighbors:
                if dist_matrix[i, j] != 0:
                    W[i] += 1 - np.log(self.k) / np.log(dist_matrix[i, j] + 1e-10)
                
        return H, W

    def compute_similarity(self, H, W):
        """Compute similarity matrix using hypergraph structure."""
        # Convert to sparse matrices
        H_sparse = sp.csr_matrix(H)
        # Compute hyperedge-based (Sh) and vertex-based (Sv) similarities
        Sh = H_sparse @ H_sparse.T
        Sv = H_sparse.T @ H_sparse
        # Elementwise product
        S = Sh.toarray() * Sv.toarray()
        return S

    def rank_images(self, query_idx, similarity_matrix):
        """Return ranked list of images based on similarity matrix.
        
        Args:
            query_idx: Index of the query image.
            similarity_matrix: Similarity matrix.
        
        Returns:
            Ranked indices (highest similarity first).
        """
        similarities = similarity_matrix[query_idx]
        ranked_idx = np.argsort(similarities)[::-1]
        return ranked_idx

    def evaluate_precision(self, ranked_idx, ground_truth, k=10):
        """Compute precision@k for ranked results."""
        if k > len(ranked_idx):
            k = len(ranked_idx)
        relevant = 0
        for i in range(k):
            if ranked_idx[i] in ground_truth:
                relevant += 1
        return relevant / k

# --- New function: Compute cosine similarity ---
def compute_cosine_similarity(features):
    """
    Compute the cosine similarity matrix for the given features.
    Args:
        features: A NumPy array of shape (N, feature_dim).
    Returns:
        A NumPy array (N x N) of cosine similarities.
    """
    # Normalize feature vectors to unit norm
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / (norms + 1e-10)
    cosine_sim_matrix = np.dot(normalized_features, normalized_features.T)
    return cosine_sim_matrix

def process_dataset(dataset_path, k=5):
    """Process a complete dataset and return the retrieval system."""
    retrieval = HypergraphImageRetrieval(k=k)
    dataset = ImageDataset(dataset_path)
    
    print("Extracting features...")
    features_list = []
    for i in range(len(dataset)):
        image_path = dataset[i]
        feat = retrieval.extract_features(image_path)
        features_list.append(feat)
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} images")
    features = np.stack(features_list)
    
    print("Building hypergraph...")
    H, W = retrieval.build_hypergraph(features)
    
    print("Computing hypergraph similarity (if needed)...")
    hypergraph_similarity = retrieval.compute_similarity(H, W)
    
    return retrieval, dataset, hypergraph_similarity, features

if __name__ == "__main__":
    # Adjust dataset path as needed.
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")
    k = 5  # Number of nearest neighbors
    
    # Process dataset: extract features and (optionally) build hypergraph
    retrieval, dataset, hypergraph_similarity, features = process_dataset(dataset_path, k)
    
    # --- Use cosine similarity instead ---
    cosine_similarity_matrix = compute_cosine_similarity(features)
    
    # You can now use the cosine similarity matrix to rank images.
    query_idx = 371  # Adjust the query index as needed
    ranked_idx = retrieval.rank_images(query_idx, cosine_similarity_matrix)
    
    print(f"\nQuery image: {dataset.get_image_name(query_idx)}")
    print("\nTop 5 similar images using cosine similarity:")
    for i in range(5):
        idx = ranked_idx[i]
        score = cosine_similarity_matrix[query_idx, idx]
        print(f"{i+1}. {dataset.get_image_name(idx)} - Score: {score:.4f}")