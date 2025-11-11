import torch
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from torch.nn import functional as F

# Folder containing generated images
image_folder = 'generated_images/'

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load Inception model
inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model.eval()
inception_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load images
images = []
for filename in os.listdir(image_folder):
    print(filename)
    if filename.endswith(('png', 'jpg', 'jpeg')):
        img = Image.open(os.path.join(image_folder, filename)).convert('RGB')
        images.append(preprocess(img).unsqueeze(0))

images = torch.cat(images, 0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
images = images.to(device)

# Compute softmax probabilities
with torch.no_grad():
    preds = F.softmax(inception_model(images), dim=1).cpu().numpy()

# Compute Inception Score
def inception_score(preds, splits=10):
    N = preds.shape[0]
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = part * (np.log(part) - np.log(py[None, :]))
        split_scores.append(np.exp(np.mean(np.sum(scores, axis=1))))
    return float(np.mean(split_scores)), float(np.std(split_scores))

mean_score, std_score = inception_score(preds)
print(f"Inception Score: {mean_score:.4f} ± {std_score:.4f}")
