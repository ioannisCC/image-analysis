import os
from pathlib import Path
import glob

class ImageDataset:
    def __init__(self, root_dir):
        """Initialize dataset from directory
        
        Args:
            root_dir: Path to directory containing images
        """
        self.root_dir = Path(root_dir)
        
        # supported image extensions
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # get all image paths
        self.image_paths = []
        for ext in self.image_extensions:
            self.image_paths.extend(glob.glob(str(self.root_dir / f'*{ext}')))
            self.image_paths.extend(glob.glob(str(self.root_dir / f'*{ext.upper()}')))
        
        # sort paths for consistency
        self.image_paths.sort()
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get path of image at index idx"""
        return self.image_paths[idx]
    
    def get_image_name(self, idx):
        """Get filename of image at index idx"""
        return os.path.basename(self.image_paths[idx])