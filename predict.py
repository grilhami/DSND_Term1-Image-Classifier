import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision

from collections import OrderedDict
from glob import glob
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("directory_to_images", help="Directory name containing images (only accept batch of imagez of size 64 or more for input; example: dir/to/imgs/).", type=str)
parser.add_argument("--topk", help="To K classes to predict.", type=int, default=5)
parser.add_argument("--category_names", help="Path to son file containing index to class name.", type=str, default="./cat_to_name.json")
parser.add_argument("--device", help="Set device to do inference on.", type=str, default="cpu")

args = parser.parse_args()

IMAGES_DIR = args.directory_to_images + "*/*"
TOP_K = args.topk
DEVICE = torch.device("cuda") if args.device == "gpu" else torch.device("cpu")

torch.cuda.empty_cache()

with open(args.category_names, 'r') as f:
    CATEGORY_NAMES = json.load(f)
    
def load_model(file_path):
    model = torchvision.models.resnet152(pretrained=True)
    checkpoint = torch.load(file_path)
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(checkpoint["input_size"], checkpoint["hidden_layers_1"])),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.2)),
                                            ('fc2', nn.Linear(checkpoint["hidden_layers_1"], checkpoint["hidden_layers_2"])),
                                            ('relu', nn.ReLU()),
                                            ('bn', nn.BatchNorm1d(checkpoint["hidden_layers_2"])),
                                            ('dropout', nn.Dropout(p=0.2)),
                                            ('fc3', nn.Linear(checkpoint["hidden_layers_2"], checkpoint["output_size"])),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.fc = classifier
    model.fc.state_dict = checkpoint["state_dict"]
    
    model.class_to_idx = checkpoint["class_to_idx"]
    return model
    
# Create function for processing images before inference
def process_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img = img.resize([255, 255])
    
    width, height = img.size # https://stackoverflow.com/a/16648197

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    img = img.crop((left, top, right, bottom))
    img = np.array(img)/255
    
    # Uncomment to see processed image
    #plt.imshow(img)
    
    img = (img - np.array(mean)) / np.array(std)
    img = img.transpose((2, 0, 1))
    
    return img

def predict(image_paths, load_model, device, topk=5, return_pred=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if not isinstance(image_paths, list):
        raise TypeError("Input of images must be a list. Not a single image.")
    
    if len(image_paths) != 64:
        raise ValueError("Length of list must be 64.")
        
    images = np.array(list(map(process_image, image_paths)))
    image = torch.from_numpy(images).type(torch.FloatTensor)
    model, image = load_model.to(device), image.to(device)
    
    with torch.no_grad():
        model.eval()
        output = model(image)
        
    ps = torch.exp(output)
    probs, classes = ps.topk(5, dim=1)
    
    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    probs = probs[return_pred].cpu().numpy().tolist()
    classes = classes[return_pred].cpu().numpy().tolist()
    
    return probs, [idx_to_class[x] for x in classes]
    # TODO: Implement the code to predict the class from an image 

if __name__ == "__main__":
    # Load model
    print("Loading model...")
    model = load_model("checkpoint_resnet.pth")
    print("###DONE###", end="\n\n")
    # Get images names
    print("Getting images...")
    images = glob(IMAGES_DIR)[:64] # 64 images for inference
    print("###DONE###", end="\n\n")
    
    # Get predictions
    print("Predicting...")
    probs, classes = predict(images, model, DEVICE, TOP_K, return_pred=TOP_K)
    labels = [CATEGORY_NAMES[idx] for idx in classes]
    
    import pandas as pd
    data = pd.DataFrame({'probability': probs, 'class': classes, 'label': labels})
    print(data)
    
    