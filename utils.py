import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import cv2


# Load and preprocess image
def load_and_preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

# Green color detection for plant image validation
def is_plant_image(image):
    image_np = np.array(image)
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv_image, (36, 25, 25), (86, 255, 255))
    green_ratio = np.sum(green_mask) / (green_mask.shape[0] * green_mask.shape[1])
    return green_ratio > 0.2  # Adjust threshold based on test results

def load_model(model_path):
    # Load VGG-11 model architecture
    model = models.vgg11()
    
    # Modify the final layer to match the number of classes (120 in this case)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=120)
    
    # Load the state dictionary (weights) with map_location to CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode
    model.eval()
    return model
