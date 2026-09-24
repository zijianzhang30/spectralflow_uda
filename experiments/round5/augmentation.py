# Functions copied verbatim from official utils.py. Preserve original axes.
import numpy as np
import torch
def radiation_noise(data, alpha_range=(0.9, 1.1), beta=0.04): #pavia/houston = 0.04
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    x = alpha * data + beta * noise
    return alpha * data + beta * noise

def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5 # True
    vertical = np.random.random() > 0.5 # False
    if horizontal:
        data = np.fliplr(data)
        data = torch.from_numpy(data.copy())
    if vertical:
        data = np.flipud(data)
        data = torch.from_numpy(data.copy())
    return data
