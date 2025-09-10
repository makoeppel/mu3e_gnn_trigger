"""
Converts ROOT files to NumPy arrays for further processing.

Functions:
- convert_root_to_npy: Extracts spacetime data from a ROOT file and saves it to a specified directory.
- get_image_slices_from_root: Extracts image slices from a ROOT file and saves them as NumPy arrays.
"""

import uproot
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
import pickle
import os


def convert_mppc_timestamp_to_ns(mppc_timestamp):
    fine_part = mppc_timestamp % 2**8
    coarse_part = mppc_timestamp // 2**8
    time_ns = fine_part * 0.05 + coarse_part * 8
    return time_ns


def convert_pixel_timestamp_to_ns(pixel_timestamp):
    time_ns = pixel_timestamp * 8
    return time_ns


def adjust_pixel_timestamps(
    timestamps: np.ndarray, padding_value: int = -999, timeframe_length=64
) -> np.ndarray:
    """
    Adjusts timestamps based on the data mask.
    If data is padded, the corresponding timestamps are set to -1.
    """
    shape = timestamps.shape
    flat_timestamps = timestamps.flatten()
    flat_mask = flat_timestamps != padding_value
    flat_timestamps[flat_mask] = convert_pixel_timestamp_to_ns(
        flat_timestamps[flat_mask]
    )
    flat_timestamps[flat_mask] = (
        flat_timestamps[flat_mask]
        - (flat_timestamps[flat_mask] // timeframe_length) * timeframe_length
    )
    adjusted_timestamps = flat_timestamps.reshape(shape)
    return adjusted_timestamps


def adjust_mppc_timestamps(
    timestamps: np.ndarray, padding_value: int = -999, timeframe_length=64
) -> np.ndarray:
    """
    Adjusts timestamps based on the data mask.
    If data is padded, the corresponding timestamps are set to -1.
    """
    shape = timestamps.shape
    converted_timestamps = np.full_like(timestamps, padding_value, dtype=np.float64)
    flat_timestamps = timestamps.flatten()
    flat_converted_timestamps = converted_timestamps.flatten()
    flat_mask = flat_timestamps != padding_value
    flat_converted_timestamps[flat_mask] = convert_mppc_timestamp_to_ns(
        flat_timestamps[flat_mask]
    )
    flat_converted_timestamps[flat_mask] = (
        flat_converted_timestamps[flat_mask]
        - (flat_converted_timestamps[flat_mask] // timeframe_length) * timeframe_length
    )
    adjusted_timestamps = flat_converted_timestamps.reshape(shape)
    return adjusted_timestamps



def reorder_nla(nla: np.ndarray, padding_value: int = -999) -> np.ndarray:
    """
    Reorders the NLA array to ensure that non-padded entries are at the beginning.
    Assumes padding is identifiable via `nla[:, :, 0] == padding_value`.
    """
    # Identify valid entries
    if nla.ndim == 3:
        valid_mask = nla[:, :, 0] != padding_value
        B, N, D = nla.shape
        flat_nla = nla.reshape(B * N, D)
        flat_valid_mask = valid_mask.reshape(B * N)

        # Get indices of valid entries
        valid_indices = np.nonzero(flat_valid_mask)[0]

        # Allocate output
        reordered_nla = np.full_like(nla, padding_value)

        counts = valid_mask.sum(axis=1)

        # Fill output using advanced indexing
        row_ids = np.repeat(np.arange(B), counts)
        group_counts = np.bincount(row_ids, minlength=B)

        # Compute start indices for placing data
        start_idx = np.zeros_like(group_counts)
        np.cumsum(group_counts[:-1], out=start_idx[1:])

        # Where to write valid entries in each row
        insert_pos = np.hstack([np.arange(c) for c in group_counts])
        reordered_nla[row_ids, insert_pos] = flat_nla[valid_indices]

    else:
        valid_mask = nla != padding_value
        B, N = nla.shape
        flat_nla = nla.reshape(B * N)
        flat_valid_mask = valid_mask.reshape(B * N)

        # Get indices of valid entries
        valid_indices = np.nonzero(flat_valid_mask)[0]

        # Allocate output
        reordered_nla = np.full_like(nla, padding_value)

        counts = valid_mask.sum(axis=1)
        # Fill output using advanced indexing
        row_ids = np.repeat(np.arange(B), counts)
        group_counts = np.bincount(row_ids, minlength=B)

        # Compute start indices for placing data
        start_idx = np.zeros_like(group_counts)
        np.cumsum(group_counts[:-1], out=start_idx[1:])

        # Where to write valid entries in each row
        insert_pos = np.hstack([np.arange(c) for c in group_counts])
        reordered_nla[row_ids, insert_pos] = flat_nla[valid_indices]

    # Compute number of valid entries per batch
    counts = valid_mask.sum(axis=1)
    # Flatten for easier fancy indexing

    return reordered_nla


def sort_by_feature(
    data: np.ndarray, feature: np.ndarray, padding_value=-1
) -> np.ndarray:
    """
    Vectorized version: Sorts `data` along axis 1 by `feature`, respecting a padding mask.
    """
    if data.shape[0:2] != feature.shape[0:2]:
        raise ValueError(
            "Data and feature must have the same number of rows and columns."
        )

    # Create a valid mask over first 2 dims
    if data.ndim == 3:
        valid_mask = (data != padding_value).any(axis=-1)  # shape (B, N)
    elif data.ndim == 2:
        valid_mask = data != padding_value
    else:
        raise ValueError("Data must be 2D or 3D array.")

    # Number of valid elements per row
    num_valid = valid_mask.sum(axis=1)

    # argsort feature, ignoring invalid entries
    sort_indices = np.argsort(np.where(valid_mask, feature, np.inf), axis=1)

    # Prepare output filled with padding_value
    sorted_data = np.full_like(data, padding_value)

    # Use np.take_along_axis to reorder data
    if data.ndim == 3:
        sorted_full = np.take_along_axis(data, sort_indices[:, :, None], axis=1)
    elif data.ndim == 2:
        sorted_full = np.take_along_axis(data, sort_indices[:, :], axis=1)
    else:
        raise ValueError("Data must be 2D or 3D array.")

    # Mask out only the valid sorted parts
    for i, n_valid in enumerate(num_valid):
        if n_valid > 0:
            sorted_data[i, :n_valid] = sorted_full[i, :n_valid]

    return sorted_data


def load_ak_series_to_numpy(
    series: pd.Series, max_cols: int = 256, fill_value: int = -999
) -> np.ndarray:
    """
    Converts an Awkward Array Series to a padded NumPy array (2D).
    Pads each row to `max_cols`, truncates longer rows, and ignores empty arrays.
    """
    # Combine series into one awkward array
    ak_array = ak.Array(series.to_list())  # safe if series contains ak Arrays or lists

    # Filter out arrays of invalid lengths
    lengths = ak.num(ak_array)
    mask = (lengths > 0) & (lengths <= max_cols)
    ak_array = ak_array[mask]

    # Clip longer arrays (optional depending on your needs)
    ak_array = ak.pad_none(ak_array, max_cols, clip=True)

    # Replace None with fill_value
    ak_array_filled = ak.fill_none(ak_array, fill_value)

    # Convert to NumPy
    result = ak.to_numpy(ak_array_filled)
    return result


def load_event_ak_to_numpy(
    series: list[pd.Series],
    cutoff: int = 256,
    fill_value: int = -999,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A list of Awkward Array Series to padded NumPy arrays.
    Pads each row to `cutoff` and `cutoff`, truncates longer rows for all series,
    and ignores empty arrays.
    """
    ak_arrays = [ak.Array(s.to_list()) for s in series]

    ak_lengths = [ak.num(ak_array) for ak_array in ak_arrays]

    ak_masks = [(lengths > 0) & (lengths <= cutoff) for lengths in ak_lengths]

    combined_mask = ak_masks[0]
    for mask in ak_masks[1:]:
        combined_mask = mask & combined_mask

    masked_arrays = [ak_array[combined_mask] for ak_array in ak_arrays]

    none_padded_arrays = [
        ak.pad_none(ak_array, cutoff, clip=True) for ak_array in masked_arrays
    ]

    filled_arrays = [
        ak.to_numpy(ak.fill_none(ak_array, fill_value))
        for ak_array in none_padded_arrays
    ]
    return filled_arrays


def convert_mppc_to_location(
    mppc: np.ndarray,
    col_index: np.ndarray,
    mppc_positions: pd.DataFrame,
    mc_hit_id: np.ndarray,
    track_id: pd.Series,
    padding_value: float = -999,
    add_layer_as_feature=False,
) -> tuple[np.ndarray[float], np.ndarray[int]]:
    """
    Converts a 1D array of MPPC IDs to their corresponding positions in space.
    The MPPC IDs are expected to be in the 'mppc' column of the mppc_positions DataFrame.
    The function returns a 2D array of positions with shape (N, 3), where N is the number of MPPCs.
    Each row corresponds to the (cx, cy, cz) coordinates of an MPPC.

    Args:
        mppc (np.ndarray): A 2D numpy array of shape (N, seq_length) containing MPPC IDs.
        col_index (np.ndarray): A 2D numpy array of shape (N, seq_length) containing column indices for each MPPC hit.
        mppc_positions (pd.DataFrame): A DataFrame containing MPPC positions with columns ['mppc', 'vx', 'vy', 'vz', 'colx', 'coly', 'colz'].
        mc_hit_id (np.ndarray): A 2D numpy array of shape (N, seq_length) containing MC hit IDs corresponding to each MPPC hit.
        track_id (pd.Series): A Series mapping MC hit IDs to track IDs.
        padding_value (float): The value used for padding in the input arrays. Defaults to -1.

    Returns:
        tuple[np.ndarray[float], np.ndarray[int]]: A tuple containing:
        - A 3D numpy array of shape (N, seq_length, 3) with MPPC positions.
        - A 2D numpy array of shape (N, seq_length) with corresponding track IDs.
    """

    # Validate input
    if not isinstance(mppc, np.ndarray) or mppc.ndim != 2:
        raise ValueError("mppc must be a 2D numpy array.")

    if mppc.shape != col_index.shape:
        raise ValueError("mppc and col_index must have the same shape.")

    required_columns = ["mppc", "vx", "vy", "vz", "colx", "coly", "colz"]

    if not all(col in mppc_positions.columns for col in required_columns):
        raise ValueError(
            f"mppc_positions DataFrame must contain the following columns: {required_columns}"
        )

    mppc_positions = mppc_positions.set_index("mppc")

    flatted_mppc = mppc.flatten()
    flat_mask = flatted_mppc != padding_value

    mppc_data = mppc_positions.loc[flatted_mppc[flat_mask]]

    flat_col_index = col_index.flatten()[flat_mask]
    mc_hit_id = mc_hit_id.flatten()[flat_mask]

    vx = mppc_data["vx"].to_numpy()
    vy = mppc_data["vy"].to_numpy()
    vz = mppc_data["vz"].to_numpy()

    col_x = mppc_data["colx"].to_numpy()
    col_y = mppc_data["coly"].to_numpy()
    col_z = mppc_data["colz"].to_numpy()

    x = vx + (128 - flat_col_index + 0.5) * col_x
    y = vy + (128 - flat_col_index + 0.5) * col_y
    z = vz + (128 - flat_col_index + 0.5) * col_z

    track_id_array = track_id[mc_hit_id].to_numpy()

    # Normalize track_id to start from 0 for each event
    track_id = np.full(mppc.shape, -1, dtype=np.int64)
    flat_track_id = track_id.reshape(-1)
    flat_track_id[flat_mask] = track_id_array

    num_features = 3
    if add_layer_as_feature:
        num_features += 1

    locations = np.full(
        (*mppc.shape, num_features), dtype=float, fill_value=padding_value
    )
    features = [x, y, z]
    if add_layer_as_feature:
        layer = np.full_like(x, 2.5, dtype=float)
        features.append(layer)
    flat_locations = locations.reshape(-1, num_features)
    flat_locations[flat_mask] = np.stack(features, axis=1)

    return locations, track_id


def convert_pid_to_location(
    pixel_id: np.ndarray,
    sensor_positions: pd.DataFrame,
    mc_hit_id: np.ndarray,
    track_id: pd.Series,
    padding_value: float = -999,
    sensor_fault_rate=0.0,
    add_layer_as_feature=False,
) -> tuple[np.ndarray[float], np.ndarray[int]]:
    if sensor_positions.empty:
        raise ValueError("sensor_positions DataFrame is empty.")
    if (track_id is not None) ^ (mc_hit_id is not None):
        raise ValueError("Both track_id and mc_hit_id must be provided together.")
    if track_id is not None and mc_hit_id.shape != pixel_id.shape:
        raise ValueError("track_id length must match pixel_id length.")

    required_columns = [
        "id",
        "vx",
        "vy",
        "vz",
        "rowx",
        "rowy",
        "rowz",
        "colx",
        "coly",
        "colz",
    ]
    if not all(col in sensor_positions.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {required_columns}")

    # Preprocessing
    sensor_positions = sensor_positions.set_index("id", drop=False)
    valid_mask = pixel_id != padding_value

    # Decode pixel_id to chip, col, row
    hit_chip_id = (pixel_id // 2**16).astype(np.int32)
    hit_col_id = ((pixel_id // 2**8) % 2**8).astype(np.float32)
    hit_row_id = (pixel_id % 2**8).astype(np.float32)

    # Flatten inputs for easier indexing
    flat_valid_mask = valid_mask.flatten()
    flat_chip_id = hit_chip_id.flatten()
    flat_col_id = hit_col_id.flatten()
    flat_row_id = hit_row_id.flatten()

    mc_hit_id = mc_hit_id.flatten()

    keep_mask = flat_chip_id // 2**12 == 0

    layer_id = ((flat_chip_id // 2**10) % 4) + 1

    if sensor_fault_rate > 0:
        keep_mask = (
            (np.random.rand(len(sensor_data)) >= sensor_fault_rate)
            | (layer_id == 1 | layer_id == 2)
        ) & keep_mask

    flat_valid_mask = flat_valid_mask & keep_mask

    # Filter only valid pixel ids
    valid_chip_ids = flat_chip_id[flat_valid_mask]
    valid_cols = flat_col_id[flat_valid_mask] + 0.5
    valid_rows = flat_row_id[flat_valid_mask] + 0.5
    valid_layer_id = (((flat_chip_id // 2**10) % 4) + 1)[flat_valid_mask]
    mc_hit_id = mc_hit_id[flat_valid_mask]

    # Lookup transformation vectors
    try:
        sensor_data = sensor_positions.loc[valid_chip_ids]
    except KeyError:
        raise ValueError("Some chip IDs not found in sensor_positions.")

    # Compute positions
    vx = sensor_data["vx"].values
    vy = sensor_data["vy"].values
    vz = sensor_data["vz"].values
    rowx = sensor_data["rowx"].values
    rowy = sensor_data["rowy"].values
    rowz = sensor_data["rowz"].values
    colx = sensor_data["colx"].values
    coly = sensor_data["coly"].values
    colz = sensor_data["colz"].values

    x = vx + valid_cols * colx + valid_rows * rowx
    y = vy + valid_cols * coly + valid_rows * rowy
    z = vz + valid_cols * colz + valid_rows * rowz
    layer = valid_layer_id.astype(np.float32)

    track_id_array = track_id[mc_hit_id].to_numpy()

    # Normalize track_id to start from 0 for each event
    track_id = np.full(pixel_id.shape, -1, dtype=np.int64)
    flat_track_id = track_id.reshape(-1)
    flat_track_id[flat_valid_mask] = track_id_array
    # Normalize: subtract per-event min where valid

    # Fill output array
    num_features = 3 + (1 if add_layer_as_feature else 0)
    features = [x, y, z]
    if add_layer_as_feature:
        features.append(layer)

    location = np.full((*pixel_id.shape, num_features), padding_value, dtype=np.float64)
    flat_location = location.reshape(-1, num_features)
    flat_location[flat_valid_mask] = np.stack(features, axis=1)

    return location, track_id


def convert_root_to_npy(
    file_path: str,
    out_dir: str,
    out_name: str,
    padding_value: float = -999,
    hit_cutoff: int = 256,
    add_layer_as_feature=False,
):
    """
    Extracts spacetime data from a ROOT file and saves it to a specified directory.
    Parameters:
    - file_path (str): Path to the ROOT file.
    - out_dir (str): Directory to save the output files.
    - out_name (str): Base name for the output files.
    - padding_value (float): Value used for padding in the output arrays. Defaults to -1.
    - hit_cutoff (int): Maximum number of hits per event. Defaults to 256.
    """
    if not file_path.endswith(".root"):
        raise ValueError("File must be a ROOT file with .root extension.")
    if not out_dir.endswith("/"):
        out_dir += "/"
    if not out_name:
        raise ValueError("Output name must be provided.")

    if padding_value == -1:
        raise Warning("Padding value of -1 may conflict with valid data. Consider using a different padding value.")

    with uproot.open(file_path) as file:
        sensor_positions = file["alignment/sensors"].arrays(library="pd")
        mppc_positions = file["alignment/mppcs"].arrays(library="pd")
        fibre_positions = file["alignment/fibres"].arrays(library="pd")
        event_data = file["mu3e"].arrays(library="pd")
        if sensor_positions.empty or mppc_positions.empty or fibre_positions.empty:
            raise ValueError("Sensor, MPPC, or fibre positions are empty.")
        if event_data.empty:
            raise ValueError("Event data is empty.")
        mc_track_id = file["mu3e_mchits"]["mc_track"].arrays(library="pd")["mc_track"]
        mc_track_truth = file["mu3e_mc_tracks"].arrays(library="pd").set_index("mother")

    mc_track_truth = mc_track_truth[["px", "py", "pz","e", "pdg"]]
    # Convert pixel and MPPC data to numpy arrays
    (
        pixel_hit_id,
        pixel_hit_timestamp,
        mppc_id,
        mppc_col_id,
        mppc_time,
        pixel_mc_hit_id,
        mppc_mc_hit_id,
    ) = load_event_ak_to_numpy(
        [
            event_data["hit_pixelid"],
            event_data["hit_timestamp"],
            event_data["fibremppc_mppc"],
            event_data["fibremppc_col"],
            event_data["fibremppc_timestamp"],
            event_data["hit_mc_i"],
            event_data["fibremppc_mc_i"],
        ],
        cutoff=hit_cutoff,
        fill_value=padding_value,
    )

    if pixel_hit_id.shape != pixel_hit_timestamp.shape:
        raise ValueError("Pixel hit ID and timestamp shapes do not match.")
    if mppc_id.shape != mppc_col_id.shape or mppc_id.shape != mppc_time.shape:
        raise ValueError("MPPC ID, column ID, and time shapes do not match.")
    if pixel_hit_id.shape[0] != mppc_id.shape[0]:
        raise ValueError("Number of events in pixel and MPPC data do not match.")

    # Convert pixel IDs to positions
    pixel_positions, pixel_track_ids = convert_pid_to_location(
        pixel_hit_id,
        sensor_positions,
        padding_value=padding_value,
        sensor_fault_rate=0.0,
        add_layer_as_feature=add_layer_as_feature,
        mc_hit_id=pixel_mc_hit_id,
        track_id=mc_track_id,
    )
    masked_pixel_hits = np.all(pixel_positions == padding_value, axis=-1)
    pixel_hit_timestamp[masked_pixel_hits] = padding_value

    pixel_hit_track_truth = np.full((*pixel_hit_id.shape, 5), padding_value, dtype=np.float32)
    pixel_hit_track_truth_flat = pixel_hit_track_truth.reshape(-1, 5)
    valid_pixel_mask = (pixel_hit_id.flatten() != padding_value) & (pixel_hit_id.flatten() != -1)
    pixel_hit_track_truth_flat[valid_pixel_mask] = mc_track_truth.iloc[mc_track_id[pixel_mc_hit_id.flatten()[valid_pixel_mask]]].to_numpy()
    pixel_hit_track_truth = pixel_hit_track_truth_flat.reshape((*pixel_hit_id.shape, 5))



    # Convert MPPC IDs to positions
    mppc_positions, mppc_track_ids = convert_mppc_to_location(
        mppc_id,
        mppc_col_id,
        mppc_positions,
        padding_value=padding_value,
        mc_hit_id=mppc_mc_hit_id,
        track_id=mc_track_id,
        add_layer_as_feature=add_layer_as_feature,
    )
    masked_mppc_hits = np.all(mppc_positions == padding_value, axis=-1)
    mppc_time[masked_mppc_hits] = padding_value

    mppc_hit_track_truth = np.full((*mppc_id.shape, 5), padding_value, dtype=np.float32)
    mppc_hit_track_truth_flat = mppc_hit_track_truth.reshape(-1, 5)
    valid_mppc_mask = (mppc_id.flatten() != padding_value) & (mppc_id.flatten() != -1)
    mppc_hit_track_truth_flat[valid_mppc_mask] = mc_track_truth.iloc[mc_track_id[mppc_mc_hit_id.flatten()[valid_mppc_mask]]].to_numpy()
    mppc_hit_track_truth = mppc_hit_track_truth_flat.reshape((*mppc_id.shape, 5))


    # Make sure track IDs start from 0 for each event
    if pixel_track_ids.shape != mppc_track_ids.shape:
        raise ValueError("Pixel and MPPC track ID shapes do not match.")
    pixel_track_ids, mppc_track_ids = remap_track_ids(pixel_track_ids, mppc_track_ids, padding_value)

    # Adjust timestamps
    pixel_hit_timestamp = adjust_pixel_timestamps(pixel_hit_timestamp, padding_value)
    mppc_time = adjust_mppc_timestamps(mppc_time, padding_value)

    pixel_hit_spacetime = np.concatenate(
        [pixel_positions, pixel_hit_timestamp[:, :, None]],
        axis=-1,
    )

    pixel_hit_track_labels = np.concatenate(
        [pixel_track_ids[:, :, None], pixel_hit_track_truth], axis=-1
    )

    mppc_hit_spacetime = np.concatenate(
        [mppc_positions, mppc_time[:, :, None]], axis=-1
    )

    mppc_hit_track_labels = np.concatenate(
        [mppc_track_ids[:, :, None], mppc_hit_track_truth], axis=-1
    )

    pixel_hit_number = (pixel_hit_spacetime != padding_value).any(axis=-1).sum(axis=-1)
    mppc_hit_number = (mppc_hit_spacetime != padding_value).any(axis=-1).sum(axis=-1)

    valid_mask = (pixel_hit_number > 0) & (mppc_hit_number > 0)

    # Apply valid mask to spacetime data
    pixel_hit_spacetime = pixel_hit_spacetime[valid_mask]
    mppc_hit_spacetime = mppc_hit_spacetime[valid_mask]

    pixel_hit_track_labels = pixel_hit_track_labels[valid_mask]
    mppc_hit_track_labels = mppc_hit_track_labels[valid_mask]

    # Save spacetime data to files
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if add_layer_as_feature:
        out_name += "_with_layer"
    pixel_file_path = f"{out_dir}{out_name}_pixel_spacetime.npy"
    mppc_file_path = f"{out_dir}{out_name}_mppc_spacetime.npy"
    pixel_track_labels_file_path = f"{out_dir}{out_name}_pixel_track_labels.npy"
    mppc_track_labels_file_path = f"{out_dir}{out_name}_mppc_track_labels.npy"
    
    np.save(pixel_file_path, pixel_hit_spacetime)
    np.save(mppc_file_path, mppc_hit_spacetime)
    np.save(pixel_track_labels_file_path, pixel_hit_track_labels)
    np.save(mppc_track_labels_file_path, mppc_hit_track_labels)
    print(
        f"Saved pixel spacetime to {pixel_file_path} with shape {pixel_hit_spacetime.shape}"
    )
    print(
        f"Saved MPPC spacetime to {mppc_file_path} with shape {mppc_hit_spacetime.shape}"
    )
    print(
        f"Saved pixel track labels to {pixel_track_labels_file_path} with shape {pixel_hit_track_labels.shape}"
    )
    print(
        f"Saved MPPC track labels to {mppc_track_labels_file_path} with shape {mppc_hit_track_labels.shape}"
    )


def remap_track_ids(pixel_track_ids, mppc_track_ids, padding_value = -999):
    maxiint = max(np.max(mppc_track_ids), np.max(pixel_track_ids)) + 1

    # Normalize track_id to start from 0 for each event
    flat_mppc_track_ids = mppc_track_ids.ravel()

    # Normalize track_id to start from 0 for each event
    flat_pixel_track_ids = pixel_track_ids.ravel()

    n_events, n_hits = mppc_track_ids.shape

    # Event index for each position
    event_idx = np.repeat(np.arange(n_events), n_hits)

    # Mask out entries that are -1 or 0
    valid_mppc_mask = (mppc_track_ids != -1) & (mppc_track_ids != padding_value)
    valid_pixel_mask = (pixel_track_ids != -1) & (pixel_track_ids != padding_value)

    flat_mppc_mask = valid_mppc_mask.ravel()
    flat_pixel_mask = valid_pixel_mask.ravel()

    # Compute per-event minima (ignore invalid entries)
    mppc_mins = np.full(n_events, maxiint, dtype=np.int64)
    pixel_mins = np.full(n_events, maxiint, dtype=np.int64)
    np.minimum.at(
        mppc_mins, event_idx[flat_mppc_mask], flat_mppc_track_ids[flat_mppc_mask]
    )
    np.minimum.at(
        pixel_mins, event_idx[flat_pixel_mask], flat_pixel_track_ids[flat_pixel_mask]
    )

    mins = np.min([mppc_mins, pixel_mins], axis=0)

    flat_mppc_track_ids[flat_mppc_mask] -= mins[event_idx[flat_mppc_mask]] - 1
    flat_pixel_track_ids[flat_pixel_mask] -= mins[event_idx[flat_pixel_mask]] - 1

    # Reshape back to original shape
    mppc_track_ids = flat_mppc_track_ids.reshape(mppc_track_ids.shape)
    pixel_track_ids = flat_pixel_track_ids.reshape(pixel_track_ids.shape)
    return pixel_track_ids, mppc_track_ids