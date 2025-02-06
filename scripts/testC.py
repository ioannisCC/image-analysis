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
        """Initialize the image retrieval system
        
        Args:
            feature_extractor: Name of pretrained model to use
            k: Number of nearest neighbors for hypergraph construction
        """
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature extractor
        if feature_extractor == 'resnet18':
            model = models.resnet18(pretrained=True)
            # Remove final classification layer
            self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
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
        """Extract deep features from an image"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(image)
        return features.cpu().squeeze().flatten().numpy()

    def build_hypergraph(self, features):
        """Construct hypergraph from feature vectors"""
        N = features.shape[0]
        
        # Compute pairwise distances efficiently using matrix operations
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist = np.linalg.norm(features - features[i], axis=1)
            dist_matrix[i] = dist
                
        # Find k-nearest neighbors
        H = np.zeros((N, N))
        for i in range(N):
            idx = np.argsort(dist_matrix[i])[:self.k]
            H[i, idx] = 1
            
        # Compute hyperedge weights
        W = np.zeros(N)
        for i in range(N):
            neighbors = np.where(H[i] > 0)[0]
            for j in neighbors:
                if dist_matrix[i,j] != 0:  # Avoid log(0)
                    W[i] += 1 - np.log(self.k) / np.log(dist_matrix[i,j] + 1e-10)
                
        return H, W

    def compute_similarity(self, H, W):
        """Compute similarity matrix using hypergraph structure"""
        # Convert to sparse matrices for efficiency
        H_sparse = sp.csr_matrix(H)
        W_diag = sp.diags(W)
        
        # Compute Sh and Sv
        Sh = H_sparse @ H_sparse.T
        Sv = H_sparse.T @ H_sparse
        
        # Convert back to numpy for element-wise multiplication
        S = Sh.toarray() * Sv.toarray()
        
        return S

    def rank_images(self, query_idx, S):
        """Return ranked list of images based on similarity matrix"""
        similarities = S[query_idx]
        ranked_idx = np.argsort(similarities)[::-1]
        return ranked_idx

    def evaluate_precision(self, ranked_idx, ground_truth, k=10):
        """Compute precision@k for ranked results"""
        if k > len(ranked_idx):
            k = len(ranked_idx)
        relevant = 0
        for i in range(k):
            if ranked_idx[i] in ground_truth:
                relevant += 1
        return relevant / k

def process_dataset(dataset_path, k=5):
    """Process a complete dataset and return the retrieval system"""
    # Initialize system
    retrieval = HypergraphImageRetrieval(k=k)
    dataset = ImageDataset(dataset_path)
    
    # Extract features
    print("Extracting features...")
    features = []
    for i in range(len(dataset)):
        image_path = dataset[i]
        feat = retrieval.extract_features(image_path)
        features.append(feat)
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} images")
    
    features = np.stack(features)
    
    # Build hypergraph
    print("Building hypergraph...")
    H, W = retrieval.build_hypergraph(features)
    
    # Compute similarities
    print("Computing similarities...")
    S = retrieval.compute_similarity(H, W)
    
    return retrieval, dataset, S

if __name__ == "__main__":
    # Example usage
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")
    k = 5  # Number of nearest neighbors
    
    # Process dataset
    retrieval, dataset, similarity_matrix = process_dataset(dataset_path, k)
    
    # Example: retrieve similar images for first image
    query_idx = 371
    ranked_idx = retrieval.rank_images(query_idx, similarity_matrix)
    
    # Print results
    print(f"\nQuery image: {dataset.get_image_name(query_idx)}")
    print("\nTop 5 similar images:")
    for i in range(5):
        idx = ranked_idx[i]
        print(f"{i+1}. {dataset.get_image_name(idx)}")