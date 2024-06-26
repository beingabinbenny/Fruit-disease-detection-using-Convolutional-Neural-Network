import torch
import torchvision.transforms as transforms
from PIL import Image

import data


classes = ['Alternaria', 'Anthracnose', 'Black_Mould_Rot', 'Stem_and_Rot', 'Healthy']

# Load the trained model
model_path = '/Users/abinbenny/Documents/adithya project/fruit/MangoFruitDDS/model/model.pth'
model = data.Net(num_classes=len(classes))
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Function to classify an image
def classify_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    
    # Use the model to predict the class
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    
    # Map the predicted class index to the class name
    predicted_class = classes[predicted.item()]
    
    return predicted_class

# Example usage
image_path = '/Users/abinbenny/Documents/adithya project/fruit/MangoFruitDDS/SenMangoFruitDDS_original/Alternaria/alternaria_003.jpg'
predicted_class = classify_image(image_path)
print(f"The image is classified as: {predicted_class}")
