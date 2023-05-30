## 1/6 basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# plt.rcParams["figure.figsize"] = (7,7)
# plt.rcParams['lines.linewidth'] = 5
# plt.rcParams['font.size'] = 20
# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=20)

plt.rcParams["figure.figsize"] = (4,4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] =8
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)


get_ipython().run_line_magic('matplotlib', 'inline') # %matplotlib inline

# get_ipython().run_line_magic('matplotlib', 'notebook') # %matplotlib notebook
# # print("Matplotlib backend is=", plt.get_backend())

## 2/6 specials
import h5py
import math
import utm
import timeit 


## 3/6 settings
import sys # print('The python version getting used is: ', sys.version)
import os # print('Current directory is: ', os.getcwd()) # print('Conda environment getting used is: ',os.environ['CONDA_DEFAULT_ENV'])

## 4/6 extras
import numpy.linalg as linalg
from PIL import Image
import re
import itertools
import urllib.request
import datetime 
import pytz
import time
import scipy
import scipy.signal as sig
import scipy.stats as stats
import scipy.constants # print("SciPy also thinks that the speed of light is c = %.1F" %scipy.constants.c ) 

import pickle as pkl
from joblib import dump, load
import copy
from copy import deepcopy
import tqdm
# from tqdm import tqdm
import itertools

from mpl_toolkits.mplot3d import Axes3D

print("Current Time =", datetime.datetime.now().strftime("%H:%M:%S"))



## 5/6 MINE! ._~
from pathlib import Path
import pdb
import seaborn as sns
import argparse 


## 6/6 ML

# import torch # ! conda install pytorch torchvision torchaudio -c pytorch -y
# # print("Torch version is", torch.__version__)
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

# from torchvision.transforms import ToTensor
# from torchvision.transforms import Lambda


# from torchvision import datasets, models, transforms

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, PolynomialFeatures, SplineTransformer
# from sklearn.pipeline import make_pipeline


# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression,LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.svm import LinearSVC, SVC, SVR

# from sklearn.multioutput import MultiOutputRegressor

# from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.model_selection import ParameterGrid

# from sklearn.manifold import TSNE


"""
 This cycle defaults to rcParams["axes.prop_cycle"] (default: cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])).

The location of the legend: 
The strings 'upper left', 'upper right', 'lower left', 'lower right' place the legend at the corresponding corner of the axes/figure.
The strings 'upper center', 'lower center', 'center left', 'center right' place the legend at the center of the corresponding edge of the axes/figure.

"""
#!/usr/bin/env python3
## Working with environment at /Users/aartisingh/.conda/envs/copr_conda_env:
# !jt --reset #t onedork -T -N -kl