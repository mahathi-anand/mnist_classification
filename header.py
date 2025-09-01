#Importing the necessary packages

#Math and system libraries
import numpy as np
import os
from termcolor import colored


#Data libraries
import pandas as pd

#Image and plotting libraries
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from PIL import Image

#Torch libraries
import torch
from torch import nn
from torch.nn.functional import one_hot
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall
#Printing Functions 

def print_error(message):
    print(colored(message, 'red'))

def print_warning(message):
    print(colored(message, 'yellow'))

def print_info(message):
    print(colored(message, 'blue'))
    
def print_success(message):
    print(colored(message, 'green'))