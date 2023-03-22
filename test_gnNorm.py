import torch

import torch.nn as nn

batch = 1
group = 2
channel = 4
height = 4
width = 4


input = torch.randn(batch, channel, height, width)

print(input, input.shape)

gn = nn.GroupNorm(int(channel / group), channel)

output = gn(input)

print(output, output.shape)
