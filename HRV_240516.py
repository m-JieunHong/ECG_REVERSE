
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import random
import pandas as pd
import neurokit2 as nk
import re
import glob
import os 
import pickle
from collections import Counter
from datetime import datetime
from scipy.spatial.distance import euclidean
import time
from dtaidistance import dtw
import mne
from lxml import etree

def read_edf(path, type='ECG'):
    edf_record = mne.io.read_raw_edf(path)
    channel_idx = edf_record.__dict__['_raw_extras'][0]['ch_names'].index(type)
    stop = edf_record.__dict__['_raw_extras'][0]['nsamples']
    signal = edf_record.get_data(picks=channel_idx, start=0, stop=stop, return_times=True)
    # print(edf_record.__dict__)
    sr = edf_record.__dict__['info']['sfreq']

    return signal, int(sr)

def write_txt(path, text, label=0):
  if label == 0:
    with open(path, 'w+') as f:
        np.savetxt(f, text)
  else:
    with open(path, 'w+') as f:
        np.savetxt(f, text, fmt="%s")

# nsrr version
def read_nsrr_xml(ann_path, id):
    event_type = []
    event_concept = []
    start = []
    duration = []
    signal_location = []
    tree = etree.parse(ann_path+id+'-nsrr.xml')
    root = tree.getroot()
    datas = root.getchildren()
    for d in datas:
        if (d.tag == 'ScoredEvents'):
            tmp = d.getchildren()
            for i in range(len(tmp)):
                for t in (tmp[i].getchildren()):
                    if t.tag == "EventType":
                        event_type.append(t.text)
                    elif t.tag == "EventConcept":
                        event_concept.append(t.text)
                    elif t.tag == "Start":
                        start.append(t.text)
                    elif t.tag == "Duration":
                        duration.append(t.text)


    return np.array(event_type), np.array(event_concept), np.array(start), np.array(duration)


def labelings_stage_nsrr(path, ann_path, id):
    signal, sr = read_edf(path)
    event_type, event_concept, start, duration = read_nsrr_xml(ann_path, id)
    arr = np.zeros(signal[0].shape[1])
    # stage 만 가져오기
    event_concept = event_concept[np.where(event_type == "Stages|Stages")]
    start = start[np.where(event_type == "Stages|Stages")]
    duration = duration[np.where(event_type == "Stages|Stages")]

    for idx in range(len(start)):
        start_ = float(start[idx])
        duration_ = float(duration[idx])
        event_idx = (event_concept[idx]).split("|")[1]
        (arr[np.where(np.logical_and(signal[1] >= start_, signal[1] < duration_+start_))]) = event_idx

    return signal, arr, sr

def rri_ver_1(rp_path):
    df = pd.read_csv(rp_path)
    r_peaks = list(map(int, df['rpointadj'].tolist()))
    rr_interval = np.diff(r_peaks)
    r_peaks.pop(0)
    return np.array(r_peaks), np.array(rr_interval)


def butter_bandpass(sr, lowcut=0.15, highcut=50, order=2):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a
    
def butter_bandpass_filter(sig, sr):
    b, a = butter_bandpass(sr)
    y = scipy.signal.lfilter(b, a, sig)
    return y

def butter_lowpass_filter(sig, sr, lowcut=0.6 , order=2):
    nyq = 0.5 * sr
    low = lowcut / nyq
    # Get the filter coefficients 
    b, a = scipy.signal.butter(order, low, btype='low', analog=False)
    y = scipy.signal.filtfilt(b, a, sig)
    return y

def labeling_apnea_nsrr(signal, ann_path, id):
    apnea_idx = []
    event_type, event_concept, start, duration = read_nsrr_xml(ann_path, id)
    arr = np.zeros(signal[0].shape[1])
        # apnea 만 가져오기
    concat_index =np.array([True if re.search("Apnea", i) else True if re.search("Hypopnea", i) else False for i in event_concept])
    start_apnea = start[concat_index]
    duration_apnea = duration[concat_index]
    
    for idx in range(len(start_apnea)):
        start_ = float(start_apnea[idx])
        duration_ = float(duration_apnea[idx])
        arr[np.where(np.logical_and(signal[1] >= start_, signal[1] < duration_+start_))] = 1
        apnea_idx.append(start_)
        apnea_idx.append(duration_)

    return arr

def rri_ver_2(signal, sr):
    peaks, _ = nk.ecg_peaks(signal, sampling_rate=sr)
    r_peaks = peaks[peaks["ECG_R_Peaks"] == 1].index.tolist()
    rr_interval = np.diff(np.array(r_peaks))
    r_peaks.pop(0)
    return np.array(r_peaks), np.array(rr_interval)

def write_txt(path, text):
    with open(path, 'wb') as f:
        np.save(f, text)

def hr_resampling(signal, sr, r_peaks, rr_interval, method="hard"):
    # rri 4Hz resampling
    new_r_peaks = np.linspace(r_peaks[0], r_peaks[-1], (len(signal) // sr) * 4)
    if method == "hard":
        f = scipy.interpolate.interp1d(r_peaks, rr_interval, kind='linear')
    elif method == "soft":
        f = scipy.interpolate.interp1d(r_peaks, rr_interval, kind='quadratic')
    new_rr_int = f(new_r_peaks)

    return np.array(new_r_peaks), np.array(new_rr_int)

def rri_filtering(rr_interval, r_peaks, margin=6, filtering_ratio=0.3):
    filtered_rri = list(rr_interval[:margin])
    filtered_rri_time = list(r_peaks[:margin])
    for interval_idx in range(margin, len(rr_interval)):
        rri_mean = np.mean(filtered_rri[-margin:])
        if rri_mean * (1 - filtering_ratio) < rr_interval[interval_idx] < rri_mean * (1 + filtering_ratio):
            filtered_rri.append(rr_interval[interval_idx])
            filtered_rri_time.append(r_peaks[interval_idx])
    return np.array(filtered_rri), np.array(filtered_rri_time)


def calculate_time_domain_features(rr_interval):
        rr_diff = np.diff(rr_interval)
        rr_diff_squared = rr_diff ** 2

        MaxNN = np.max(rr_interval)
        MinNN = np.min(rr_interval)
        MedianNN = np.median(rr_interval)
        MeanNN = np.mean(rr_interval)
        MadNN = np.median(np.abs(rr_interval - MedianNN))
        MCVNN = MadNN / MedianNN
        SDNN = np.std(rr_interval)
        RMSSD = np.sqrt(np.mean(rr_diff_squared))
        SDSD = np.std(rr_diff)
        SDRMSSD = SDNN / RMSSD
        CVNN = SDNN / MeanNN
        CVSD = RMSSD / MeanNN
        NN50 = np.sum(np.abs(rr_diff) > 50)
        NN20 = np.sum(np.abs(rr_diff) > 20)
        pNN50 = np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100
        pNN20 = np.sum(np.abs(rr_diff) > 20) / len(rr_diff) * 100
        Prc20NN = np.percentile(rr_interval, 20)
        Prc80NN = np.percentile(rr_interval, 80)
        IQRNN = np.percentile(rr_interval, 75) - np.percentile(rr_interval, 25)

        features = {'MaxNN': MaxNN, 'MinNN': MinNN, 'MedianNN': MedianNN, 'MeanNN': MeanNN, 'MadNN': MadNN, 'MCVNN': MCVNN,
                    'SDNN': SDNN, 'RMSSD': RMSSD, 'SDSD': SDSD, 'SDRMSSD': SDRMSSD, 'CVNN': CVNN, 'CVSD': CVSD,  'NN50': NN50,
                    'NN20': NN20, 'pNN50': pNN50, 'pNN20': pNN20, 'Prc20NN': Prc20NN, 'Prc80NN': Prc80NN, 'IQRNN': IQRNN}

        return features

def calculate_frequency_domain_features(rr_interval, sr=4):
        f, pxx = scipy.signal.welch(rr_interval, fs=sr, nperseg=len(rr_interval), noverlap=0, scaling='density')

        vlf_band = (0.0033, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        vhf_band = (0.4, 0.5)

        VLF = np.trapz(pxx[(f >= vlf_band[0]) & (f < vlf_band[1])], f[(f >= vlf_band[0]) & (f < vlf_band[1])])
        LF = np.trapz(pxx[(f >= lf_band[0]) & (f < lf_band[1])], f[(f >= lf_band[0]) & (f < lf_band[1])])
        HF = np.trapz(pxx[(f >= hf_band[0]) & (f < hf_band[1])], f[(f >= hf_band[0]) & (f < hf_band[1])])
        VHF = np.trapz(pxx[(f >= vhf_band[0]) & (f < vhf_band[1])], f[(f >= vhf_band[0]) & (f < vhf_band[1])])
        TP = np.trapz(pxx, f)
        LF_HF = LF / HF
        LFn = LF / TP
        HFn = HF / TP
        LnHF = np.log(HF) if HF > 0 else 0

        features = {'VLF': VLF, 'LF': LF, 'HF': HF, 'VHF': VHF, 'TP': TP, 'LF/HF': LF_HF, 'LFn': LFn, 'HFn': HFn, 'LnHF': LnHF}

        return features

def calculate_time_domain_features(rr_interval):
        rr_diff = np.diff(rr_interval)
        rr_diff_squared = rr_diff ** 2

        MaxNN = np.max(rr_interval)
        MinNN = np.min(rr_interval)
        MedianNN = np.median(rr_interval)
        MeanNN = np.mean(rr_interval)
        MadNN = np.median(np.abs(rr_interval - MedianNN))
        MCVNN = MadNN / MedianNN
        SDNN = np.std(rr_interval)
        RMSSD = np.sqrt(np.mean(rr_diff_squared))
        SDSD = np.std(rr_diff)
        SDRMSSD = SDNN / RMSSD
        CVNN = SDNN / MeanNN
        CVSD = RMSSD / MeanNN
        NN50 = np.sum(np.abs(rr_diff) > 50)
        NN20 = np.sum(np.abs(rr_diff) > 20)
        pNN50 = np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100
        pNN20 = np.sum(np.abs(rr_diff) > 20) / len(rr_diff) * 100
        Prc20NN = np.percentile(rr_interval, 20)
        Prc80NN = np.percentile(rr_interval, 80)
        IQRNN = np.percentile(rr_interval, 75) - np.percentile(rr_interval, 25)

        features = {'MaxNN': MaxNN, 'MinNN': MinNN, 'MedianNN': MedianNN, 'MeanNN': MeanNN, 'MadNN': MadNN, 'MCVNN': MCVNN,
                    'SDNN': SDNN, 'RMSSD': RMSSD, 'SDSD': SDSD, 'SDRMSSD': SDRMSSD, 'CVNN': CVNN, 'CVSD': CVSD,  'NN50': NN50,
                    'NN20': NN20, 'pNN50': pNN50, 'pNN20': pNN20, 'Prc20NN': Prc20NN, 'Prc80NN': Prc80NN, 'IQRNN': IQRNN}

        return features



def calculate_nonlinear_domain_features(rr_interval):
    rr_x = rr_interval[:-1]
    rr_y = rr_interval[1:]

    SD1 = np.sqrt(np.std(rr_y - rr_x) / 2)
    SD2 = np.sqrt(np.std(rr_y + rr_x) / 2)
    SD1_SD2 = SD1 / SD2
    S = np.pi * SD1 * SD2
    CSI = SD2 / SD1
    CVI = np.log(SD1 * SD2)
    CSI_Modified = (SD1 ** 2) / SD2
    Stress_Index = np.sum(np.abs(rr_y - rr_x))

    def _phi(m, r):
        x = np.array([rr_interval[i:i + m] for i in range(len(rr_interval) - m + 1)])
        C = np.sum([np.sum(np.abs(xi - x) <= r, axis=1) / (len(rr_interval) - m + 1.0) for xi in x], axis=0) / len(x)
        return np.sum(np.log(C)) / len(x)

    def _count_matches(m, r):
        count = 0
        for i in range(len(rr_interval) - m):
            for j in range(i + 1, len(rr_interval) - m):
                if np.all(np.abs(rr_interval[i:i + m] - rr_interval[j:j + m]) <= r):
                    count += 1
        return count

    m = 2
    r = 0.2

    ApEn = np.abs(_phi(m + 1, r) - _phi(m, r))

    N = len(rr_interval)
    B = _count_matches(m, r) / (N - m)
    A = _count_matches(m + 1, r) / (N - m - 1)

    SampEn = -np.log(A / B) if B > 0 else float('inf')

    features = {'SD1': SD1, 'SD2': SD2, 'SD1/SD2': SD1_SD2, 'S': S, 'CSI': CSI, 'CVI': CVI, 'CSI_Modified': CSI_Modified,
                'Stress_Index': Stress_Index, 'ApEn': ApEn, 'SampEn': SampEn}

    return features

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def save_dict(path, dict_):
    with open(path,'wb') as fw:
        pickle.dump(dict_, fw)
