from HRV_240516 import read_edf
import os 
import glob
import matplotlib.pyplot as plt
from reverse_detection import *

if __name__ == "__main__":
    path = "/Users/aimmo-aip-0168/Desktop/shhs/edf"
    print(os.path.join(path, "*.edf"))
    reversed = "we don't know"
    edf_list = glob.glob(os.path.join(path, "*.edf"))
    for i in edf_list:
        signal, sr = read_edf(i)
        pattern = "shhs2\\-[0-9]{6}"
        id = re.findall(pattern, i)[0]
        r_peaks = rri_ver_2(signal[0][0])
        ensemble = []
        for idx in range(len(r_peaks)):
            rpeaks = r_peaks[idx]
            temp = median_filter(signal[0][0][rpeaks-80:rpeaks+80])
            ensemble.append(temp)
        
                # ax.plot(temp, alpha=0.2)
            if idx == 100:
                break

        
        arr = np.array(ensemble)
        avg = np.average(arr, axis=0)

        q, p_min, p_max = p_detection(avg)
        print(np.abs(avg[80]), np.abs(avg[q+40]), avg[p_min+20])
        pred_reverse = reverse_caculate(avg, q, p_min, p_max)

        plot_pq(id, reversed, pred_reverse, temp, q, p_min, p_max)

            
