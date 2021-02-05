#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, embed_size, filters, m_word=21, window_size=5):
        super(CNN, self).__init__()

        self.filters=filters
        self.conv = nn.Conv1d(in_channels=embed_size,out_channels=filters, kernel_size=window_size)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=m_word-window_size+1)

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv(x_reshaped)
        x_conv_out = self.max_pool(self.relu(x_conv))
        return x_conv_out.reshape(-1,self.filters)

### END YOUR CODE

