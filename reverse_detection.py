
import matplotlib.pyplot as plt
import glob
import numpy as np
import re 
import neurokit2 as nk
import time 
from tqdm import tqdm
import pandas as pd
from scipy.signal import medfilt

def rri_ver_2(signal, sr=250):
    peaks, _ = nk.ecg_peaks(signal, sampling_rate=sr)
    r_peaks = peaks[peaks["ECG_R_Peaks"] == 1].index.tolist()
    r_peaks.pop(0)
    return np.array(r_peaks)

def p_detection(temp):
    q = np.argmin(temp[40:80]) 
    p_min = np.argmin(temp[20:q+30])
    p_max = np.argmax(temp[20:q+30])
    return q, p_min, p_max

def reverse_caculate(signal, q, p_min, p_max):
    # if signal[p_min+20] > 0:
    #     title = 0
    if np.abs(signal[p_min+20]) >= np.abs(signal[p_max+20]):
        if np.abs(signal[80]) > np.abs(signal[q+40]):
            title = 0
        else:
            title = 1
    else:
        title = 0
    return title 

def median_filter(signal):
    xx1 = medfilt(signal, 29)
    baseline = medfilt(xx1, 99)
    return signal - baseline


def plot_pq(id, reverse, pred_reverse, temp, q, p_min, p_max):
    plt.title(f"{id}/{pred_reverse}")
    plt.plot(temp, color="black", linewidth=2)
    plt.scatter(q+40, temp[q+40], color="red", label="q")
    plt.scatter(p_min+20, temp[p_min+20], color="purple", label="p min")
    plt.scatter(p_max+20, temp[p_max+20], color="yellow", label="p max")
    plt.legend()
    plt.show()
   
def compute_gradients(x, y):
    gradients = np.gradient(y, x)
    return gradients
                
            




        





