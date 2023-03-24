import torch

import torch.nn as nn

batch = 1
group = 1
channel = 2
height = 2
width = 2


# input = torch.randn(batch, channel, height, width)
input = torch.tensor([[[[0.9021, -0.0838], [0.1884, 0.6099]], [[-1.1973, 0.6271], [-1.9284, -1.4174]]]])

sum = input.sum()
mean = input.mean()
std = input.std()
print("input sum : ", sum)
print("input mean : ", mean)
print("input std: ", std)

print("input : \n", input, input.shape)

gn = nn.GroupNorm(group, channel)

output = gn(input)

sum = output.sum()
mean = output.mean()
std = output.std()
print("output sum : ", sum)
print("output mean : ", mean)
print("output std: ", std)

print("output : \n", output, output.shape)
