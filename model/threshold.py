from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import numpy as np
from matplotlib import pyplot as plt

class Threshold:

    def __init__(self, start=0, end=100, low=0., high=1.):
        super().__init__()
        self.start = start
        self.end = end
        self.low = low
        self.high = high

    def __call__(self, x, reverse=False):
        assert x >= self.start and x < self.end
        y = self.compute(x / (self.end - self.start))
        if reverse: y = 1 - y
        return y * (self.high - self.low) + self.low

    def plot(self, reverse=False):
        xs = list(range(self.start, self.end))
        ys = [self.__call__(x, reverse=reverse) for x in xs]
        plt.plot([self.start] + xs + [self.end], 
            [self.low if not reverse else self.high] + ys +
            [self.high if not reverse else self.low])


class SigmoidThreshold(Threshold):

    def __init__(self, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def compute(self, x):
        y = (x - 0.5) / (self.alpha / 5)
        return 1 / (1 + math.exp(-y))
