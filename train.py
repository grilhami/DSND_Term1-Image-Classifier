import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser()

parser.add_argument("data_directory", help="Directory that has training, validation, and testing set.")
parser.add_argument("--save_dir", help="Directory to save checkpoints.", default=".")
parser.add_argument("--arch", help="Pick model to train on.", choices=["vgg19_bn", "resnet152", "densenet161"], default="resnet152")
parser.add_argument("--learning_rate", help="Set learning rate for training.", type=float, default=0.01)
parser.add_argument("--hidden_units_1", help="Set hidden units for first layer of fully connected", type=int, default=1024)
parser.add_argument("--hidden_units_2", help="Set hidden units for second layer of fully connected", type=int, default=512)
parser.add_argument("--epochs", help="Set epochs for training.", type=int, default=20)
parser.add_argument("--device", help="Set device to train on.", type=str, default="cpu")


args = parser.parse_args()

# Get parse value
DATA_DIR = args.data_directory
SAVE_DIR = args.save_dir
MODEL = args.arch
LEARNING_RATE = args.learning_rate
N_HIDDEN_1 = args.hidden_units_1
N_HIDDEN_2 = args.hidden_units_2
EPOCHS = args.epochs
DEVICE = torch.device("cuda") if args.device == "gpu" else torch.device("cpu")


print("Loading ddata...")
# Load data
train_dir = DATA_DIR + '/train'
valid_dir = DATA_DIR + '/valid'
test_dir = DATA_DIR + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
training_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                          transforms.RandomResizedCrop(224), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
val_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(val_datasets, batch_size=64)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
print("#"*50)
print("Training, validation, and testing data are loaded.")
print("#"*50, end="\n\n")

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Get pretrained model
print("Getting pretrained model...")
if MODEL == "resnet152":
    model = models.resnet152(pretrained=True)
elif MODEL == "vgg19_bn":
    model = models.vgg19_bn(pretrained=True)
elif MODEL == "densenet161":
    model = models.densenet161(pretrained=True)
else:
    print("Model unsupported.")
    
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, N_HIDDEN_1)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(p=0.2)),
                                        ('fc2', nn.Linear(N_HIDDEN_1, N_HIDDEN_2)),
                                        ('relu', nn.ReLU()),
                                        ('bn', nn.BatchNorm1d(N_HIDDEN_2)),
                                        ('dropout', nn.Dropout(p=0.2)),
                                        ('fc3', nn.Linear(N_HIDDEN_2, len(list(cat_to_name.values())))),
                                        ('output', nn.LogSoftmax(dim=1))]))

# Set fully connected layer
if MODEL == "resnet152":
    model.fc = classifier
else:
    model.classifier = classifier
    
# Set loss and optimizer
criterion = nn.NLLLoss()
if MODEL == "resnet152":
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
else:
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

## TODO: Use a pretrained model to classify the cat and dog images
model = model.to(DEVICE)
train_losses, val_losses = [], []

print("Training model...")
for e in range(EPOCHS):
    running_loss = 0
    for images, labels in trainloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        val_loss, accuracy = 0, 0

        with torch.no_grad():
            model.eval()
            for images, labels in valloader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                output = model(images)

                val_loss += criterion(output, labels).item()

                ps = torch.exp(output)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
        model.train()
        train_losses.append(running_loss/len(trainloader))
        val_losses.append(val_loss/len(valloader))

        print("Epoch: {}/{}".format(e+1, EPOCHS), 
              "Train losses: {:.3f}".format(train_losses[-1]),
              "Validation losses: {:.3f}".format(val_losses[-1]),
              "Validation accuracy {:.2f}".format(accuracy/len(valloader)))
        
# TODO: Save the checkpoint 
model.cpu()
model.class_to_idx = train_datasets.class_to_idx
checkpoint_cpu = {
    'batch_size': trainloader.batch_size,
    'input_size': model.fc.fc1.in_features, 
    'output_size': len(train_datasets.class_to_idx), 
    'hidden_layers_1': model.fc.fc1.out_features,
    'hidden_layers_2': model.fc.fc2.out_features,
    'state_dict': model.fc.state_dict(),
    'class_to_idx': model.class_to_idx,
    'optim_state_dict': optimizer.state_dict(),
    'epochs': EPOCHS}
torch.save(checkpoint_cpu, 'checkpoint_resnet.pth')

