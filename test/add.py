import torch

device = torch.device("cuda:0") 
left = torch.zeros(100, device=device, requires_grad=True)
right = torch.zeros(100, device=device, requires_grad=True)
grad = torch.zeros(100, device=device)

for _ in range(10):
    output = torch.add(left, right)
    output = torch.add(output, output)
    output.backward(grad)

