import torch

device = torch.device("cpu") 
left = torch.zeros(100, device=device, requires_grad=True)
right = torch.zeros(100, device=device, requires_grad=True)
grad = torch.zeros(100, device=device)

for _ in range(10):
    output = torch.add(left, right)
    output.backward(grad)
