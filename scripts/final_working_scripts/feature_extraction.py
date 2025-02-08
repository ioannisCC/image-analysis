from torchvision.models import resnet50
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import pickle


# path to 'list.txt' file
list_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "annotations", "list.txt")

image_info = []

with open(list_file, 'r') as file:
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

# load the pre-trained model
model = resnet50(pretrained=True)

# evaluate the model
model.eval()

# remove the last layer (the classification layer, fc)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

# apply the same transformations as the ones used during training
# https://pytorch.org/vision/0.18/models/generated/torchvision.models.resnet50.html

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_features(image_path, model, transform):
    try:
        img = Image.open(image_path).convert('RGB') # ensure RGB format
        image_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model(image_tensor)
        return features.squeeze(0).numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# training images
image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "images")

all_features = []

# for each entry in the data frame
for _, row in image_info_df.iterrows(): # iterrows() allow to iterate over the rows of a df
    image_name = row['image_id'] + '.jpg'
    image_path = os.path.join(image_dir, image_name)
    
    # extract features
    features = extract_features(image_path, model, transform)
    
    if features is not None:
        all_features.append({
            'features': features,
            'image_id': row['image_id'],
            'class_id': row['class_id'],
            'species': row['species'],
            'breed_id': row['breed_id']
        })
    else:
        print(f'Image {image_name} has no extracted features')

# convert list to numpy array
features_array = np.array([item['features'] for item in all_features])

print("Number of all feature entries: ", len(all_features))
print("Features shape: ", features_array.shape)
print("Number of feature entries: ", len(all_features))
if len(all_features) > 0:
    print("Shape of a single feature vector: ", all_features[0]['features'].shape)

for feature in all_features:
    feature['features'] = feature['features'].reshape(-1)  # flatten to 1d dictionary

print("Sample metadata with reshaped features:", all_features[0])
print(len(all_features))
print("Shape of reshaped feature vector:", all_features[0]['features'].shape)

# --- save the features ---

features_array = features_array.reshape(features_array.shape[0], -1)  # flatten the 1x1 spatial dimension
print(features_array.shape)  # (nr of images, nr of features for each image)

# absolute path of the current script
current_script_path = os.path.abspath(__file__)
print(f"Current script location: {current_script_path}")

# absolute path for artifacts folder
artifacts_path = os.path.abspath(os.path.join(os.path.dirname(current_script_path), '..', '..', 'artifacts'))
print(f"Attempting to save to: {artifacts_path}")

# where the files will be saved
os.makedirs(artifacts_path, exist_ok=True)  # create it if it does not exist

features_file = os.path.join(artifacts_path, 'features.npy')
all_features_file = os.path.join(artifacts_path, 'all_features.pkl')

# save files
np.save(features_file, features_array)
with open(all_features_file, 'wb') as f:
    pickle.dump(all_features, f)

print("all_features saved successfully!")