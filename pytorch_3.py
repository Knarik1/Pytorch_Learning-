# Imports
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fully connected
class NN(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x
    
# device    
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 

# dataset
train_dataset = datasets.MNIST(train=True, root="dataset/", transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(train=False, root="dataset/", transform=transforms.ToTensor(), download=True)

train_dataloader = 
    
# Create model    
model = NN(20, 10)
print(model)    
pprint.pprint(vars(model))