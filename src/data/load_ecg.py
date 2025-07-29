# src/data/load_ecg.py
import numpy as np
import struct
import os
from typing import Tuple, List, Dict

def load_signal(dat_file: str) -> np.ndarray:
    """
    Loads ECG signal from .dat file in MIT 212 format (two channels).
    
    Args:
        dat_file: Path to .dat file.
    
    Returns:
        Numpy array of shape (num_samples, 2) for two channels (e.g., MLII, V5).
    """
    with open(dat_file, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint8)
    
    num_samples = len(raw_data) // 3 * 2  # Two samples per 3 bytes
    samples = np.zeros(num_samples, dtype=np.int16)
    i = 0
    j = 0
    while j < num_samples:
        if i + 2 >= len(raw_data):
            break
        byte1, byte2, byte3 = raw_data[i:i+3]
        # First sample
        sample1 = (byte1 << 4) | (byte2 >> 4)
        if sample1 >= 2048:
            sample1 -= 4096
        samples[j] = sample1
        j += 1
        if j >= num_samples:
            break
        # Second sample
        sample2 = ((byte2 & 0x0F) << 8) | byte3
        if sample2 >= 2048:
            sample2 -= 4096
        samples[j] = sample2
        j += 1
        i += 3
    
    # Reshape to (num_samples // 2, 2) for two channels
    return samples.reshape(-1, 2)

def load_annotations(atr_file: str) -> List[Dict[str, any]]:
    """
    Parses .atr annotation file in MIT format.
    
    Args:
        atr_file: Path to .atr file.
    
    Returns:
        List of dicts with 'sample' (int), 'ann_code' (int), 'subtype' (int), etc.
    """
    annotations = []
    current_sample = 0
    subtyp = 0
    chan = 0
    num = 0
    with open(atr_file, 'rb') as f:
        while True:
            byte_pair = f.read(2)
            if len(byte_pair) != 2:
                break
            value = struct.unpack('<H', byte_pair)[0]
            ann_code = (value >> 10) & 0x3F
            interval = value & 0x3FF
            
            if ann_code == 59:  # SKIP
                if interval == 0:
                    interval_bytes = f.read(4)
                    if len(interval_bytes) != 4:
                        break
                    interval = struct.unpack('<i', interval_bytes)[0]  # Signed int
                current_sample += interval
                continue
            
            elif ann_code == 60:  # NUM
                num = interval
                continue
            
            elif ann_code == 61:  # SUB
                subtyp = interval
                continue
            
            elif ann_code == 62:  # CHN
                chan = interval
                continue
            
            elif ann_code == 63:  # AUX
                aux_bytes = f.read(interval)
                if len(aux_bytes) != interval:
                    break
                aux = aux_bytes.decode('ascii').rstrip('\x00')  # Strip nulls
                # Attach AUX to previous ann if needed; here, add as separate
                annotations.append({'sample': current_sample, 'ann_code': ann_code, 'subtype': subtyp, 'chan': chan, 'num': num, 'aux': aux})
                if interval % 2 == 1:  # Skip extra null if odd
                    f.read(1)
                continue
            
            elif ann_code == 0 and interval == 0:
                break
            
            # Regular annotation
            current_sample += interval
            annotations.append({'sample': current_sample, 'ann_code': ann_code, 'subtype': subtyp, 'chan': chan, 'num': num, 'aux': None})
    
    return annotations

def get_all_records(data_dir: str = 'data/mitbih') -> List[str]:
    """
    Gets list of record IDs (e.g., '100') from .dat files in data_dir.
    """
    records = [os.path.basename(f)[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]
    return sorted(records)

# For local testing/synthetic data (remove in prod; use for unit tests)
if __name__ == '__main__':
    # Synthetic ECG: simple sine for demo
    t = np.linspace(0, 10, 3600)
    synthetic_signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)
    synthetic_signal = synthetic_signal.reshape(-1, 1)  # Single channel
    print(synthetic_signal.shape)
