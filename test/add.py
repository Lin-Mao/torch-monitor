import torch

device = torch.device("cuda:0") 
input = torch.rand(100, device=device)
output = torch.add(input, input)

