# src/data/preprocess.py
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict
from src.data.load_ecg import load_signal, load_annotations, get_all_records

LABEL_MAP = {
    1: 0,   # N (normal) -> 0
    2: 0,   # L (left bundle) -> 0 (normal-ish)
    3: 0,   # R (right bundle) -> 0
    4: 1,   # A (atrial prem) -> 1 (S)
    5: 1,   # a (aberrant atrial) -> 1
    6: 1,   # J (nodal prem) -> 1
    7: 1,   # S (supraventricular prem) -> 1
    8: 2,   # V (ventricular prem) -> 2 (V)
    9: 2,   # F (fusion) -> 3? Wait, F is 10
    # Full map per ecgcodes.h; AAMI: N=0, S=1 (A,a,J,S,e,j), V=2 (V,E), F=3 (F), Q=4 (/,Q,? )
    10: 3,  # F (fusion normal+vent)
    11: 2,  # / (paced)
    12: 4,  # Q (unclassified)
    13: 2,  # E (vent escape)
    # Add more as needed; ignore non-beat ann_codes (>50)
}

def extract_beats(signal: np.ndarray, annotations: List[Dict[str, any]], window_size: int = 360, channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts normalized beat windows around R-peaks from annotations.
    
    Args:
        signal: (num_samples, num_channels) array.
        annotations: List from load_annotations.
        window_size: Total samples per beat (e.g., 360).
        channel: ECG channel to use (0: MLII).
    
    Returns:
        X: (num_beats, window_size, 1) float32 array.
        y: (num_beats,) int array (0-4).
    """
    half_window = window_size // 2
    X = []
    y = []
    for ann in annotations:
        if ann['ann_code'] not in LABEL_MAP or ann['sample'] < half_window or ann['sample'] + half_window > len(signal):
            continue  # Skip non-beat or out-of-bounds
        beat = signal[ann['sample'] - half_window : ann['sample'] + half_window, channel]
        # Normalize
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
        X.append(beat[:, np.newaxis])  # (360, 1)
        y.append(LABEL_MAP[ann['ann_code']])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def create_client_dataset(record_id: str, data_dir: str = 'data/mitbih', batch_size: int = 32) -> tf.data.Dataset:
    """
    Creates tf.Dataset for one client (record).
    """
    dat_file = os.path.join(data_dir, f'{record_id}.dat')
    atr_file = os.path.join(data_dir, f'{record_id}.atr')
    signal = load_signal(dat_file)
    anns = load_annotations(atr_file)
    X, y = extract_beats(signal, anns)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset.shuffle(1000).batch(batch_size)

def create_federated_dataset(data_dir: str = 'data/mitbih') -> Dict[str, tf.data.Dataset]:
    """
    Creates dict of client_id to tf.Dataset for TFF simulation.
    """
    records = get_all_records(data_dir)
    federated_data = {}
    for rec in records:
        federated_data[rec] = create_client_dataset(rec, data_dir)
    return federated_data

# Example usage (for testing with synthetic; real requires data)
if __name__ == '__main__':
    # Synthetic test
    synthetic_signal = np.random.randn(10000, 2)
    synthetic_anns = [{'sample': 500, 'ann_code': 1, 'subtype': 0, 'chan': 0, 'num': 0, 'aux': None}]
    X, y = extract_beats(synthetic_signal, synthetic_anns)
    print(X.shape, y)  # Should be (1, 360, 1), [0]
    