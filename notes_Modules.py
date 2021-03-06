import torch
import torch.nn as nn

import pprint

class MyModule(nn.Module):
    def __init__(self, in_feautures, out_features):
        super().__init__()
        torch.seed()
        self.tens = torch.Tensor([2,3])
        self.weight = nn.Parameter(torch.rand(in_feautures, out_features))
        self.fc1 = nn.Linear(in_feautures, out_features)
        self.bias = nn.Parameter(torch.rand(1, out_features), requires_grad=False)
        
    def forward(self, data):
        # x = torch.matmul(data, self.weight) + self.bias
        x = data @ self.weight + self.bias
        return x   
        
        
        
model = MyModule(3,2)
# output = model.forward(torch.rand(10,3))
# print(model.weight)
# print(MyModule)
# print("=======================")
pprint.pprint(vars(model))   
# print(list(model.parameters(recurse=False)))     

# print("================================================")
# pprint.pprint(vars(model._modules))



        