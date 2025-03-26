## 1/5 basics ###############################################################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

## 2/5 signal processing ###############################################################

import scipy.signal as sig
from scipy.signal import iirfilter, lfilter, find_peaks
import scipy.constants

from numpy.polynomial import Polynomial as PNOM


## 3/5 timezoning ###############################################################

import pytz
from datetime import datetime  # has (1. strptime that has a function called .timestamp() / 2. strftime / 3. fromtimestamp) 

import utm
import time

## 5/5 helpful ###############################################################

from pathlib import Path
import pickle as pkl
import argparse
import h5py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import math


## 4/5 ML ###############################################################

## 4/5 mine ###############################################################
import pdb

## 4/5 specials ###############################################################

## 4/5 trash ###############################################################

## 4/5 extras ###############################################################
import random

all_BS_coords = {
  'hospital': (40.77105, -111.83712),  
  'ustar': (40.76899, -111.84167),
  'bes': (40.76134, -111.84629),
  'honors': (40.76440, -111.83695),
  'smt': (40.76740, -111.83118),
}