import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

path = r'C:\Users\damia\Documents\CPdsai EEG\Preprocessed_EEG\Preprocessed_EEG'
list_mat = os.listdir(path)
list_mat.remove('readme.txt')

labels = loadmat(path + '\\' + list_mat[-1])
labels['label']

loado = list_mat[0]
print(loado)
data_o = loadmat(path +'\\' + loado)
print(data_o.keys())

eegsig1 = data_o['ww_eeg1']
plt.psd(eegsig1[1, :])
plt.show()


eegsig1 = data_o['ww_eeg1']
print(eegsig1)
print(eegsig1.shape)
plt.plot(eegsig1[1, :])
plt.show()

print(eegsig1)

columns = pd.read_excel('channel-order.xlsx',header=None)
df_list = columns.T.values.tolist()
