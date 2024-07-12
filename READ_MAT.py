import mat73
import numpy as np


def load_mat(path):
    mat_file_name = path
    mat_file = mat73.loadmat(mat_file_name)
    return mat_file

def read_ecg(file):
    return (file['dECG']-8192)/1000

def read_sr(file):
    return file['fs']

def read_rpeaks(file):
    rpeaks =  np.where(file['Rpk_label'] == 1)[0]
    return rpeaks

def read_lost(file): 
    return np.where(file['data_lost'] != True)
    



