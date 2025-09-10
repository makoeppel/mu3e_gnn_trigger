import numpy as np


def load_numpy_files(file_prefix):
    pixel_spacetime = np.load(f"{file_prefix}_pixel_spacetime.npy", allow_pickle=True)
    mppc_spacetime = np.load(f"{file_prefix}_mppc_spacetime.npy", allow_pickle=True)
    pixel_track_labels = np.load(f"{file_prefix}_pixel_track_labels.npy", allow_pickle=True)
    mppc_track_labels = np.load(f"{file_prefix}_mppc_track_labels.npy", allow_pickle=True)

    return pixel_spacetime, mppc_spacetime, pixel_track_labels, mppc_track_labels
