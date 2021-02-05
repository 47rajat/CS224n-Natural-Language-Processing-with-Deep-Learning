#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, embed_size):
        super(Highway, self).__init__()

        self.proj = nn.Linear(in_features=embed_size,out_features=embed_size,bias=True)
        self.gate = nn.Linear(in_features=embed_size,out_features=embed_size,bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        x_proj = self.relu(self.proj(x_conv_out))
        x_gate = self.sigmoid(self.gate(x_conv_out))
        x_highway = x_proj*x_gate + (1 - x_gate)*x_conv_out
        return x_highway

### END YOUR CODE 

