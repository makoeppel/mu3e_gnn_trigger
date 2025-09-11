"""
ROOT files to NumPy arrays converter.

This module provides functions to extract and process spacetime data from ROOT files,
converting them to NumPy arrays for machine learning applications.
"""

import os
import warnings
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import uproot
import awkward as ak


# Constants
DEFAULT_PADDING_VALUE = -999
DEFAULT_HIT_CUTOFF = 256
DEFAULT_TIMEFRAME_LENGTH = 64
MPPC_CENTER_OFFSET = 128
PIXEL_OFFSET = 0.5
MPPC_LAYER_ID = 2.5

# Bit manipulation constants for pixel ID decoding
CHIP_ID_SHIFT = 16
COL_ID_SHIFT = 8
BIT_MASK_8 = 2**8
LAYER_SHIFT = 10
LAYER_MASK = 4
FRAME_LENGTH_8NS = 8


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


def convert_mppc_timestamp_to_ns(mppc_timestamp: np.ndarray, padding_value = DEFAULT_PADDING_VALUE) -> np.ndarray:
    """Convert MPPC timestamp to nanoseconds."""
    valid_mask = mppc_timestamp == padding_value
    converted_mppc_timestamp = np.full_like(mppc_timestamp, padding_value, dtype=np.float32)
    flat_converted_mppc_timestamp = converted_mppc_timestamp.flatten()
    mppc_timestamp_flat = mppc_timestamp.flatten()
    fine_part = mppc_timestamp_flat % BIT_MASK_8
    coarse_part = mppc_timestamp_flat // BIT_MASK_8
    flat_converted_mppc_timestamp[~valid_mask.flatten()] = (coarse_part[~valid_mask.flatten()] % FRAME_LENGTH_8NS) * 8 + fine_part[~valid_mask.flatten()] * 0.05
    return flat_converted_mppc_timestamp.reshape(mppc_timestamp.shape)


def convert_pixel_timestamp_to_ns(pixel_timestamp: np.ndarray, padding_value = DEFAULT_PADDING_VALUE) -> np.ndarray:
    """Convert pixel timestamp to nanoseconds."""
    valid_mask = pixel_timestamp == padding_value
    converted_pixel_timestamp = np.full_like(pixel_timestamp, padding_value, dtype=np.float32)
    flat_converted_mppc_timestamp = converted_pixel_timestamp.flatten()
    pixel_timestamp_flat = pixel_timestamp.flatten()
    flat_converted_mppc_timestamp[~valid_mask.flatten()] = (pixel_timestamp_flat[~valid_mask.flatten()] % FRAME_LENGTH_8NS) * 8
    return flat_converted_mppc_timestamp.reshape(pixel_timestamp.shape)

def validate_array_shapes(*arrays) -> None:
    """Validate that all arrays have the same shape."""
    if len(arrays) < 2:
        return

    first_shape = arrays[0].shape
    for i, arr in enumerate(arrays[1:], 1):
        if arr.shape != first_shape:
            raise DataProcessingError(
                f"Array {i} shape {arr.shape} doesn't match first array shape {first_shape}"
            )


def reorder_array_by_validity(
    data: np.ndarray, padding_value: int = DEFAULT_PADDING_VALUE
) -> np.ndarray:
    """
    Reorder array to ensure non-padded entries come first.

    Args:
        data: Input array (2D or 3D)
        padding_value: Value used for padding

    Returns:
        Reordered array with valid entries first
    """
    if data.size == 0:
        return data

    if data.ndim == 3:
        valid_mask = data[:, :, 0] != padding_value
    elif data.ndim == 2:
        valid_mask = data != padding_value
    else:
        raise DataProcessingError(
            f"Unsupported array dimensions: {data.ndim}. Expected 2D or 3D."
        )

    batch_size = data.shape[0]
    result = np.full_like(data, padding_value)

    for i in range(batch_size):
        valid_indices = np.where(valid_mask[i])[0]
        if len(valid_indices) > 0:
            result[i, : len(valid_indices)] = data[i, valid_indices]

    return result


def sort_by_feature(
    data: np.ndarray, feature: np.ndarray, padding_value: int = DEFAULT_PADDING_VALUE
) -> np.ndarray:
    """
    Sort data along axis 1 by feature values, preserving padding structure.

    Args:
        data: Data array to sort
        feature: Feature array to sort by
        padding_value: Value used for padding

    Returns:
        Sorted data array
    """
    validate_array_shapes(data[:, :, 0] if data.ndim == 3 else data, feature)

    # Create validity mask
    if data.ndim == 3:
        valid_mask = (data != padding_value).any(axis=-1)
    elif data.ndim == 2:
        valid_mask = data != padding_value
    else:
        raise DataProcessingError(f"Unsupported data dimensions: {data.ndim}")

    # Sort indices, putting invalid entries at the end
    sort_indices = np.argsort(np.where(valid_mask, feature, np.inf), axis=1)

    # Apply sorting
    if data.ndim == 3:
        sorted_data = np.take_along_axis(data, sort_indices[:, :, None], axis=1)
    else:
        sorted_data = np.take_along_axis(data, sort_indices, axis=1)

    # Ensure invalid entries remain invalid
    result = np.full_like(data, padding_value)
    num_valid = valid_mask.sum(axis=1)

    for i, n_valid in enumerate(num_valid):
        if n_valid > 0:
            result[i, :n_valid] = sorted_data[i, :n_valid]

    return result


def load_awkward_to_numpy(
    series_list: List[pd.Series],
    max_length: int = DEFAULT_HIT_CUTOFF,
    fill_value: int = DEFAULT_PADDING_VALUE,
) -> List[np.ndarray]:
    """
    Convert list of Awkward Array Series to padded NumPy arrays.

    Args:
        series_list: List of pandas Series containing awkward arrays
        max_length: Maximum sequence length (longer sequences are clipped)
        fill_value: Value used for padding

    Returns:
        List of padded NumPy arrays
    """
    if not series_list:
        raise DataProcessingError("Empty series list provided")

    # Convert to awkward arrays
    ak_arrays = [ak.Array(series.to_list()) for series in series_list]

    # Calculate lengths and create combined validity mask
    lengths_list = [ak.num(ak_array) for ak_array in ak_arrays]
    valid_masks = [(lengths > 0) & (lengths <= max_length) for lengths in lengths_list]

    # Combine all validity masks
    combined_mask = valid_masks[0]
    for mask in valid_masks[1:]:
        combined_mask = combined_mask & mask

    # Apply mask and padding
    filtered_arrays = [ak_array[combined_mask] for ak_array in ak_arrays]
    padded_arrays = [
        ak.pad_none(ak_array, max_length, clip=True) for ak_array in filtered_arrays
    ]

    # Convert to NumPy
    numpy_arrays = [
        ak.to_numpy(ak.fill_none(ak_array, fill_value)) for ak_array in padded_arrays
    ]

    return numpy_arrays


def decode_pixel_ids(
    pixel_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode pixel IDs into chip, column, row, and layer components.

    Args:
        pixel_ids: Array of pixel IDs

    Returns:
        Tuple of (chip_ids, col_ids, row_ids, layer_ids)
    """
    chip_ids = (pixel_ids // (2**CHIP_ID_SHIFT)).astype(np.int32)
    col_ids = ((pixel_ids // (2**COL_ID_SHIFT)) % BIT_MASK_8).astype(np.float32)
    row_ids = (pixel_ids % BIT_MASK_8).astype(np.float32)
    layer_ids = ((chip_ids // (2**LAYER_SHIFT)) % LAYER_MASK) + 1

    return chip_ids, col_ids, row_ids, layer_ids


def convert_pixels_to_locations(
    pixel_ids: np.ndarray,
    sensor_positions: pd.DataFrame,
    mc_hit_ids: Optional[np.ndarray] = None,
    track_ids: Optional[pd.Series] = None,
    padding_value: float = DEFAULT_PADDING_VALUE,
    sensor_fault_rate: float = 0.0,
    add_layer_feature: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel IDs to 3D spatial locations.

    Args:
        pixel_ids: Array of pixel IDs
        sensor_positions: DataFrame with sensor position data
        mc_hit_ids: Monte Carlo hit IDs (optional)
        track_ids: Series mapping hit IDs to track IDs (optional)
        padding_value: Value used for padding
        sensor_fault_rate: Rate of sensor faults to simulate
        add_layer_feature: Whether to add layer as a feature

    Returns:
        Tuple of (locations, track_ids)
    """
    # Validate inputs
    if sensor_positions.empty:
        raise DataProcessingError("Sensor positions DataFrame is empty")

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
    missing_columns = [
        col for col in required_columns if col not in sensor_positions.columns
    ]
    if missing_columns:
        raise DataProcessingError(f"Missing required columns: {missing_columns}")

    if (mc_hit_ids is not None) != (track_ids is not None):
        raise DataProcessingError(
            "Both mc_hit_ids and track_ids must be provided together or both None"
        )

    if mc_hit_ids is not None and mc_hit_ids.shape != pixel_ids.shape:
        raise DataProcessingError("mc_hit_ids shape must match pixel_ids shape")

    # Setup
    sensor_positions = sensor_positions.set_index("id", drop=False)
    valid_mask = pixel_ids != padding_value

    # Decode pixel IDs
    chip_ids, col_ids, row_ids, layer_ids = decode_pixel_ids(pixel_ids)

    # Create sensor fault mask
    fault_mask = np.ones_like(valid_mask, dtype=bool)
    if sensor_fault_rate > 0:
        # Apply sensor faults, but never to layers 1 and 2
        protected_layers = (layer_ids == 1) | (layer_ids == 2)
        fault_mask = (
            np.random.rand(*chip_ids.shape) >= sensor_fault_rate
        ) | protected_layers

    # Filter valid chip IDs (top 4 bits must be 0)
    chip_valid_mask = (chip_ids // (2**12)) == 0
    combined_mask = valid_mask & fault_mask & chip_valid_mask

    # Get valid data
    valid_chip_ids = chip_ids[combined_mask]
    valid_cols = col_ids[combined_mask]
    valid_rows = row_ids[combined_mask]
    valid_layer_ids = layer_ids[combined_mask].astype(np.float32)

    # Lookup sensor data
    try:
        sensor_data = sensor_positions.loc[valid_chip_ids]
    except KeyError as e:
        raise DataProcessingError(f"Some chip IDs not found in sensor_positions: {e}")

    # Calculate 3D positions
    positions = {
        "x": (
            sensor_data["vx"].values
            + valid_cols * sensor_data["colx"].values
            + valid_rows * sensor_data["rowx"].values
        ),
        "y": (
            sensor_data["vy"].values
            + valid_cols * sensor_data["coly"].values
            + valid_rows * sensor_data["rowy"].values
        ),
        "z": (
            sensor_data["vz"].values
            + valid_cols * sensor_data["colz"].values
            + valid_rows * sensor_data["rowz"].values
        ),
    }

    # Prepare features
    features = [positions["x"], positions["y"], positions["z"]]
    if add_layer_feature:
        features.append(valid_layer_ids)

    # Create output arrays
    num_features = len(features)
    locations = np.full(
        (*pixel_ids.shape, num_features), padding_value, dtype=np.float64
    )
    flat_locations = locations.reshape(-1, num_features)
    flat_combined_mask = combined_mask.flatten()
    flat_locations[flat_combined_mask] = np.stack(features, axis=1)

    # Handle track IDs
    result_track_ids = np.full(pixel_ids.shape, -1, dtype=np.int64)
    if mc_hit_ids is not None and track_ids is not None:
        valid_mc_hits = mc_hit_ids[combined_mask]
        track_id_values = track_ids[valid_mc_hits].to_numpy()
        flat_track_ids = result_track_ids.flatten()
        flat_track_ids[flat_combined_mask] = track_id_values
        result_track_ids = flat_track_ids.reshape(pixel_ids.shape)

    return locations, result_track_ids


def convert_mppc_to_locations(
    mppc_ids: np.ndarray,
    col_indices: np.ndarray,
    mppc_positions: pd.DataFrame,
    mc_hit_ids: Optional[np.ndarray] = None,
    track_ids: Optional[pd.Series] = None,
    padding_value: float = DEFAULT_PADDING_VALUE,
    add_layer_feature: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MPPC IDs to 3D spatial locations.

    Args:
        mppc_ids: Array of MPPC IDs
        col_indices: Array of column indices
        mppc_positions: DataFrame with MPPC position data
        mc_hit_ids: Monte Carlo hit IDs (optional)
        track_ids: Series mapping hit IDs to track IDs (optional)
        padding_value: Value used for padding
        add_layer_feature: Whether to add layer as a feature

    Returns:
        Tuple of (locations, track_ids)
    """
    # Validate inputs
    validate_array_shapes(mppc_ids, col_indices)

    required_columns = ["mppc", "vx", "vy", "vz", "colx", "coly", "colz"]
    missing_columns = [
        col for col in required_columns if col not in mppc_positions.columns
    ]
    if missing_columns:
        raise DataProcessingError(f"Missing required columns: {missing_columns}")

    if (mc_hit_ids is not None) != (track_ids is not None):
        raise DataProcessingError(
            "Both mc_hit_ids and track_ids must be provided together or both None"
        )

    # Setup
    mppc_positions = mppc_positions.set_index("mppc")
    valid_mask = mppc_ids != padding_value

    # Get valid data
    flat_mppc_ids = mppc_ids.flatten()
    flat_col_indices = col_indices.flatten()
    flat_valid_mask = valid_mask.flatten()

    valid_mppc_ids = flat_mppc_ids[flat_valid_mask]
    valid_col_indices = flat_col_indices[flat_valid_mask]

    # Lookup MPPC data
    try:
        mppc_data = mppc_positions.loc[valid_mppc_ids]
    except KeyError as e:
        raise DataProcessingError(f"Some MPPC IDs not found in mppc_positions: {e}")

    # Calculate 3D positions
    positions = {
        "x": (
            mppc_data["vx"].values
            + (MPPC_CENTER_OFFSET - valid_col_indices + PIXEL_OFFSET)
            * mppc_data["colx"].values
        ),
        "y": (
            mppc_data["vy"].values
            + (MPPC_CENTER_OFFSET - valid_col_indices + PIXEL_OFFSET)
            * mppc_data["coly"].values
        ),
        "z": (
            mppc_data["vz"].values
            + (MPPC_CENTER_OFFSET - valid_col_indices + PIXEL_OFFSET)
            * mppc_data["colz"].values
        ),
    }

    # Prepare features
    features = [positions["x"], positions["y"], positions["z"]]
    if add_layer_feature:
        layer_values = np.full_like(positions["x"], MPPC_LAYER_ID, dtype=np.float32)
        features.append(layer_values)

    # Create output arrays
    num_features = len(features)
    locations = np.full(
        (*mppc_ids.shape, num_features), padding_value, dtype=np.float64
    )
    flat_locations = locations.reshape(-1, num_features)
    flat_locations[flat_valid_mask] = np.stack(features, axis=1)

    # Handle track IDs
    result_track_ids = np.full(mppc_ids.shape, -1, dtype=np.int64)
    if mc_hit_ids is not None and track_ids is not None:
        flat_mc_hit_ids = mc_hit_ids.flatten()
        valid_mc_hits = flat_mc_hit_ids[flat_valid_mask]
        track_id_values = track_ids[valid_mc_hits].to_numpy()
        flat_track_ids = result_track_ids.flatten()
        flat_track_ids[flat_valid_mask] = track_id_values
        result_track_ids = flat_track_ids.reshape(mppc_ids.shape)

    return locations, result_track_ids


def remap_track_ids_to_zero_indexed(
    pixel_track_ids: np.ndarray,
    mppc_track_ids: np.ndarray,
    padding_value: int = DEFAULT_PADDING_VALUE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remap track IDs to start from 0 for each event.

    Args:
        pixel_track_ids: Pixel track ID array
        mppc_track_ids: MPPC track ID array
        padding_value: Value used for padding

    Returns:
        Tuple of remapped (pixel_track_ids, mppc_track_ids)
    """
    validate_array_shapes(pixel_track_ids, mppc_track_ids)

    n_events, n_hits = pixel_track_ids.shape

    # Create masks for valid entries
    pixel_valid_mask = (pixel_track_ids != -1) & (pixel_track_ids != padding_value)
    mppc_valid_mask = (mppc_track_ids != -1) & (mppc_track_ids != padding_value)

    # Find minimum track ID for each event
    pixel_mins = np.full(n_events, np.inf)
    mppc_mins = np.full(n_events, np.inf)

    for i in range(n_events):
        if np.any(pixel_valid_mask[i]):
            pixel_mins[i] = np.min(pixel_track_ids[i, pixel_valid_mask[i]])
        if np.any(mppc_valid_mask[i]):
            mppc_mins[i] = np.min(mppc_track_ids[i, mppc_valid_mask[i]])

    # Use the minimum across both detector types for each event
    combined_mins = np.minimum(pixel_mins, mppc_mins)
    combined_mins[np.isinf(combined_mins)] = 0  # Handle events with no valid tracks

    # Apply offset to make track IDs start from 0
    pixel_result = pixel_track_ids.copy()
    mppc_result = mppc_track_ids.copy()

    for i in range(n_events):
        if combined_mins[i] != np.inf:
            pixel_result[i, pixel_valid_mask[i]] -= (combined_mins[i] - 1).astype(
                np.int64
            )
            mppc_result[i, mppc_valid_mask[i]] -= (combined_mins[i] - 1).astype(
                np.int64
            )

    return pixel_result.astype(float), mppc_result.astype(float)


def convert_root_to_numpy(
    file_path: str,
    output_dir: str,
    output_name: str,
    padding_value: float = DEFAULT_PADDING_VALUE,
    hit_cutoff: int = DEFAULT_HIT_CUTOFF,
    add_layer_feature: bool = False,
    n_events: Optional[int] = None,
) -> None:
    """
    Convert ROOT file to NumPy arrays for machine learning.

    Args:
        file_path: Path to input ROOT file
        output_dir: Directory to save output files
        output_name: Base name for output files
        padding_value: Value used for padding
        hit_cutoff: Maximum hits per event
        add_layer_feature: Whether to include layer information
    """
    # Validate inputs
    if not file_path.endswith(".root"):
        raise DataProcessingError("Input file must have .root extension")

    if not output_name:
        raise DataProcessingError("Output name must be provided")

    if padding_value == -1:
        warnings.warn(
            "Padding value of -1 may conflict with valid data. "
            "Consider using a different padding value.",
            UserWarning,
        )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load data from ROOT file
        with uproot.open(file_path) as file:
            sensor_positions = file["alignment/sensors"].arrays(library="pd")
            mppc_positions = file["alignment/mppcs"].arrays(library="pd")
            fibre_positions = file["alignment/fibres"].arrays(library="pd")
            if n_events is not None:
                event_data = file["mu3e"].arrays(library="pd", entry_stop=n_events)
            else:
                event_data = file["mu3e"].arrays(library="pd")
            mc_track_ids = file["mu3e_mchits"]["mc_track"].arrays(library="pd")[
                "mc_track"
            ]
            mc_track_truth = (
                file["mu3e_mc_tracks"].arrays(library="pd").set_index("mother")
            )

        # Validate loaded data
        if any(
            df.empty
            for df in [sensor_positions, mppc_positions, fibre_positions, event_data]
        ):
            raise DataProcessingError("One or more required data tables are empty")

        mc_track_truth = mc_track_truth[["px", "py", "pz", "e", "pdg"]]

        # Convert awkward arrays to numpy
        data_arrays = load_awkward_to_numpy(
            [
                event_data["hit_pixelid"],
                event_data["hit_timestamp"],
                event_data["fibremppc_mppc"],
                event_data["fibremppc_col"],
                event_data["fibremppc_timestamp"],
                event_data["hit_mc_i"],
                event_data["fibremppc_mc_i"],
            ],
            max_length=hit_cutoff,
            fill_value=padding_value,
        )

        (
            pixel_ids,
            pixel_timestamps,
            mppc_ids,
            mppc_cols,
            mppc_times,
            pixel_mc_hits,
            mppc_mc_hits,
        ) = data_arrays

        # Convert IDs to spatial positions
        pixel_locations, pixel_track_ids = convert_pixels_to_locations(
            pixel_ids,
            sensor_positions,
            pixel_mc_hits,
            mc_track_ids,
            padding_value,
            add_layer_feature=add_layer_feature,
        )

        mppc_locations, mppc_track_ids = convert_mppc_to_locations(
            mppc_ids,
            mppc_cols,
            mppc_positions,
            mppc_mc_hits,
            mc_track_ids,
            padding_value,
            add_layer_feature=add_layer_feature,
        )
        # Convert timestamps to nanoseconds
        pixel_timestamps = convert_pixel_timestamp_to_ns(pixel_timestamps, padding_value)
        mppc_times = convert_mppc_timestamp_to_ns(mppc_times, padding_value)

        # Mask timestamps for invalid positions
        pixel_invalid_mask = np.all(pixel_locations == padding_value, axis=-1)
        mppc_invalid_mask = np.all(mppc_locations == padding_value, axis=-1)
        pixel_timestamps[pixel_invalid_mask] = padding_value
        mppc_times[mppc_invalid_mask] = padding_value

        # Remap track IDs to start from 0
        pixel_track_ids, mppc_track_ids = remap_track_ids_to_zero_indexed(
            pixel_track_ids, mppc_track_ids, padding_value
        )

        # Create spacetime arrays
        pixel_spacetime = np.concatenate(
            [pixel_locations, pixel_timestamps[..., None]], axis=-1
        )
        mppc_spacetime = np.concatenate(
            [mppc_locations, mppc_times[..., None]], axis=-1
        )

        # Create track truth arrays
        def create_track_truth(mc_hit_ids, track_ids_array, data_shape):
            track_truth = np.full((*data_shape, 5), padding_value, dtype=np.float32)
            flat_truth = track_truth.reshape(-1, 5)
            valid_mask = (mc_hit_ids.flatten() != padding_value) & (
                mc_hit_ids.flatten() != -1
            )

            if np.any(valid_mask):
                truth_data = mc_track_truth.iloc[
                    track_ids_array[mc_hit_ids.flatten()[valid_mask]]
                ].to_numpy()
                flat_truth[valid_mask] = truth_data

            return track_truth

        pixel_track_truth = create_track_truth(
            pixel_mc_hits, mc_track_ids, pixel_ids.shape
        )
        mppc_track_truth = create_track_truth(
            mppc_mc_hits, mc_track_ids, mppc_ids.shape
        )

        # Combine track IDs and truth
        pixel_track_labels = np.concatenate(
            [pixel_track_ids[..., None], pixel_track_truth], axis=-1
        )
        mppc_track_labels = np.concatenate(
            [mppc_track_ids[..., None], mppc_track_truth], axis=-1
        )

        # Filter events with valid hits
        pixel_hit_counts = (pixel_spacetime != padding_value).any(axis=-1).sum(axis=-1)
        mppc_hit_counts = (mppc_spacetime != padding_value).any(axis=-1).sum(axis=-1)
        valid_events = (pixel_hit_counts > 0) & (mppc_hit_counts > 0)

        # Apply filter
        pixel_spacetime = pixel_spacetime[valid_events]
        mppc_spacetime = mppc_spacetime[valid_events]
        pixel_track_labels = pixel_track_labels[valid_events]
        mppc_track_labels = mppc_track_labels[valid_events]

        # Save results
        file_paths = {
            "pixel_spacetime": f"{output_dir}/{output_name}_pixel_spacetime.npy",
            "mppc_spacetime": f"{output_dir}/{output_name}_mppc_spacetime.npy",
            "pixel_labels": f"{output_dir}/{output_name}_pixel_track_labels.npy",
            "mppc_labels": f"{output_dir}/{output_name}_mppc_track_labels.npy",
        }

        np.save(file_paths["pixel_spacetime"], pixel_spacetime)
        np.save(file_paths["mppc_spacetime"], mppc_spacetime)
        np.save(file_paths["pixel_labels"], pixel_track_labels)
        np.save(file_paths["mppc_labels"], mppc_track_labels)

        # Print summary
        print(f"Successfully processed {file_path}")
        print(
            f"Pixel spacetime: {pixel_spacetime.shape} -> {file_paths['pixel_spacetime']}"
        )
        print(
            f"MPPC spacetime: {mppc_spacetime.shape} -> {file_paths['mppc_spacetime']}"
        )
        print(
            f"Pixel labels: {pixel_track_labels.shape} -> {file_paths['pixel_labels']}"
        )
        print(f"MPPC labels: {mppc_track_labels.shape} -> {file_paths['mppc_labels']}")

    except Exception as e:
        raise DataProcessingError(
            f"Failed to process ROOT file {file_path}: {str(e)}"
        ) from e
