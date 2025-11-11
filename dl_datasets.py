import os
import tarfile
import urllib.request
import pickle
from PIL import Image
import numpy as np

# URL for CIFAR-10 Python version
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
dataset_dir = "datasets/cifar10"

# Download the dataset
if not os.path.exists("cifar-10-python.tar.gz"):
    print("Downloading CIFAR-10 dataset...")
    urllib.request.urlretrieve(url, "cifar-10-python.tar.gz")
    print("Download complete.")

# Extract the dataset
if not os.path.exists("cifar-10-batches-py"):
    print("Extracting dataset...")
    with tarfile.open("cifar-10-python.tar.gz") as tar:
        tar.extractall()
    print("Extraction complete.")

# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to load a batch
def load_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        # Reshape and convert to uint8 images
        images = data.reshape(-1, 3, 32, 32).transpose(0,2,3,1)
    return images, labels

# Function to save images
def save_images(images, labels, split):
    for idx, (img, label) in enumerate(zip(images, labels)):
        class_name = class_names[label]
        dir_path = os.path.join(dataset_dir, split, class_name)
        os.makedirs(dir_path, exist_ok=True)
        img_path = os.path.join(dir_path, f"{split}_{idx}.png")
        Image.fromarray(img).save(img_path)
    print(f"{split} images saved!")

# Load all training batches
all_images = []
all_labels = []
for i in range(1, 6):
    batch_file = f"cifar-10-batches-py/data_batch_{i}"
    images, labels = load_batch(batch_file)
    all_images.append(images)
    all_labels.extend(labels)

all_images = np.concatenate(all_images)
all_labels = np.array(all_labels)

# Shuffle the data
np.random.seed(42)  # For reproducibility
indices = np.arange(len(all_images))
np.random.shuffle(indices)
all_images = all_images[indices]
all_labels = all_labels[indices]

# Split into 80% train, 20% val
split_idx = int(0.8 * len(all_images))
train_images, val_images = all_images[:split_idx], all_images[split_idx:]
train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

# Save train and val images
save_images(train_images, train_labels, 'train')
save_images(val_images, val_labels, 'val')

# Process test batch
test_images, test_labels = load_batch("cifar-10-batches-py/test_batch")
save_images(test_images, test_labels, 'test')

print("All images saved in 'datasets/cifar10' folder.")

