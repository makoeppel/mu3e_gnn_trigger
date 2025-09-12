"""
ROOT files to NumPy arrays converter.

This module provides functions to extract and process spacetime data from ROOT files,
converting them to NumPy arrays for machine learning applications.
"""

import os
import warnings
from typing import Tuple, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import awkward as ak


# ============================================================================
# CONSTANTS
# ============================================================================

# Default values
DEFAULT_PADDING_VALUE = -999
DEFAULT_HIT_CUTOFF = 256
DEFAULT_TIMEFRAME_LENGTH = 64

# Physical constants
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

# Required DataFrame columns
SENSOR_POSITION_COLUMNS = ["id", "vx", "vy", "vz", "rowx", "rowy", "rowz", "colx", "coly", "colz"]
MPPC_POSITION_COLUMNS = ["mppc", "vx", "vy", "vz", "colx", "coly", "colz"]
MC_TRACK_TRUTH_COLUMNS = ["px", "py", "pz", "e", "pdg"]


# ============================================================================
# EXCEPTIONS
# ============================================================================

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_file_path(file_path: Union[str, Path]) -> Path:
    """Validate input file path."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise DataProcessingError(f"Input file does not exist: {file_path}")
    
    if file_path.suffix != ".root":
        raise DataProcessingError("Input file must have .root extension")
    
    return file_path


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str], name: str) -> None:
    """Validate DataFrame has required columns."""
    if df.empty:
        raise DataProcessingError(f"{name} DataFrame is empty")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataProcessingError(f"{name} missing required columns: {missing_columns}")


def validate_array_shapes(*arrays: np.ndarray) -> None:
    """Validate that all arrays have the same shape."""
    if len(arrays) < 2:
        return
    
    first_shape = arrays[0].shape
    for i, arr in enumerate(arrays[1:], 1):
        if arr.shape != first_shape:
            raise DataProcessingError(
                f"Array {i} shape {arr.shape} doesn't match first array shape {first_shape}"
            )


def validate_optional_arrays(arr1: Optional[np.ndarray], arr2: Optional[np.ndarray], 
                           base_array: np.ndarray, names: Tuple[str, str]) -> None:
    """Validate optional arrays are provided together and have correct shape."""
    if (arr1 is not None) != (arr2 is not None):
        raise DataProcessingError(
            f"Both {names[0]} and {names[1]} must be provided together or both None"
        )
    
    if arr1 is not None and arr1.shape != base_array.shape:
        raise DataProcessingError(f"{names[0]} shape must match base array shape")


# ============================================================================
# TIMESTAMP CONVERSION
# ============================================================================

def convert_mppc_timestamp_to_ns(mppc_timestamp: np.ndarray, 
                                padding_value: float = DEFAULT_PADDING_VALUE) -> np.ndarray:
    """
    Convert MPPC timestamp to nanoseconds.
    
    Args:
        mppc_timestamp: Array of MPPC timestamps
        padding_value: Value used for invalid/padding entries
        
    Returns:
        Converted timestamps in nanoseconds
    """
    valid_mask = mppc_timestamp != padding_value
    result = np.full_like(mppc_timestamp, padding_value, dtype=np.float32)
    
    # Vectorized conversion for valid timestamps
    valid_timestamps = mppc_timestamp[valid_mask]
    fine_part = valid_timestamps % BIT_MASK_8
    coarse_part = valid_timestamps // BIT_MASK_8
    
    result[valid_mask] = (
        (coarse_part % FRAME_LENGTH_8NS) * 8 + fine_part * 0.05
    )
    
    return result


def convert_pixel_timestamp_to_ns(pixel_timestamp: np.ndarray, 
                                 padding_value: float = DEFAULT_PADDING_VALUE) -> np.ndarray:
    """
    Convert pixel timestamp to nanoseconds.
    
    Args:
        pixel_timestamp: Array of pixel timestamps
        padding_value: Value used for invalid/padding entries
        
    Returns:
        Converted timestamps in nanoseconds
    """
    valid_mask = pixel_timestamp != padding_value
    result = np.full_like(pixel_timestamp, padding_value, dtype=np.float32)
    
    # Vectorized conversion for valid timestamps
    valid_timestamps = pixel_timestamp[valid_mask]
    result[valid_mask] = (valid_timestamps % FRAME_LENGTH_8NS) * 8
    
    return result


# ============================================================================
# AWKWARD ARRAY PROCESSING
# ============================================================================

def load_awkward_to_numpy(series_list: List[pd.Series],
                         max_length: int = DEFAULT_HIT_CUTOFF,
                         fill_value: int = DEFAULT_PADDING_VALUE) -> List[np.ndarray]:
    """
    Convert list of Awkward Array Series to padded NumPy arrays.
    
    Args:
        series_list: List of pandas Series containing awkward arrays
        max_length: Maximum sequence length (longer sequences are dropped)
        fill_value: Value used for padding
        
    Returns:
        List of padded NumPy arrays, aligned across all input series
    """
    if not series_list:
        raise DataProcessingError("Empty series list provided")
    
    # Convert to awkward arrays
    ak_arrays = [ak.Array(series.to_list()) for series in series_list]
    
    # Calculate lengths and create combined validity mask
    lengths_list = [ak.num(ak_array, axis=1) for ak_array in ak_arrays]
    valid_masks = [(lengths > 0) & (lengths <= max_length) for lengths in lengths_list]
    combined_mask = np.logical_and.reduce(valid_masks)
    
    # Process arrays
    numpy_arrays = []
    for ak_array in ak_arrays:
        filtered = ak_array[combined_mask]
        padded = ak.pad_none(filtered, max_length, clip=True, axis=1)
        numpy_arrays.append(ak.to_numpy(ak.fill_none(padded, fill_value)))
    
    return numpy_arrays


# ============================================================================
# PIXEL ID PROCESSING
# ============================================================================

def decode_pixel_ids(pixel_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def create_sensor_fault_mask(chip_ids: np.ndarray, layer_ids: np.ndarray, 
                            fault_rate: float) -> np.ndarray:
    """Create mask for sensor faults, protecting layers 1 and 2."""
    if fault_rate <= 0:
        return np.ones_like(chip_ids, dtype=bool)
    
    protected_layers = (layer_ids == 1) | (layer_ids == 2)
    fault_mask = (np.random.rand(*chip_ids.shape) >= fault_rate) | protected_layers
    
    return fault_mask


# ============================================================================
# SPATIAL COORDINATE CONVERSION
# ============================================================================

def convert_pixels_to_locations(pixel_ids: np.ndarray,
                               sensor_positions: pd.DataFrame,
                               mc_hit_ids: Optional[np.ndarray] = None,
                               track_ids: Optional[pd.Series] = None,
                               padding_value: float = DEFAULT_PADDING_VALUE,
                               sensor_fault_rate: float = 0.0,
                               add_layer_feature: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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
    validate_dataframe_columns(sensor_positions, SENSOR_POSITION_COLUMNS, "sensor_positions")
    validate_optional_arrays(mc_hit_ids, track_ids, pixel_ids, ("mc_hit_ids", "track_ids"))
    
    # Setup
    sensor_positions_indexed = sensor_positions.set_index("id", drop=False)
    valid_mask = pixel_ids != padding_value
    
    # Decode pixel IDs
    chip_ids, col_ids, row_ids, layer_ids = decode_pixel_ids(pixel_ids)
    
    # Create combined validity mask
    fault_mask = create_sensor_fault_mask(chip_ids, layer_ids, sensor_fault_rate)
    chip_valid_mask = (chip_ids // (2**12)) == 0  # Top 4 bits must be 0
    combined_mask = valid_mask & fault_mask & chip_valid_mask
    
    # Extract valid data
    valid_indices = combined_mask.flatten()
    valid_chip_ids = chip_ids.flatten()[valid_indices]
    valid_cols = col_ids.flatten()[valid_indices]
    valid_rows = row_ids.flatten()[valid_indices]
    valid_layer_ids = layer_ids.flatten()[valid_indices].astype(np.float32)
    
    # Lookup sensor data
    try:
        sensor_data = sensor_positions_indexed.loc[valid_chip_ids]
    except KeyError as e:
        raise DataProcessingError(f"Some chip IDs not found in sensor_positions: {e}")
    
    # Calculate 3D positions using vectorized operations
    positions_x = (sensor_data["vx"].values + 
                  valid_cols * sensor_data["colx"].values + 
                  valid_rows * sensor_data["rowx"].values)
    
    positions_y = (sensor_data["vy"].values + 
                  valid_cols * sensor_data["coly"].values + 
                  valid_rows * sensor_data["rowy"].values)
    
    positions_z = (sensor_data["vz"].values + 
                  valid_cols * sensor_data["colz"].values + 
                  valid_rows * sensor_data["rowz"].values)
    
    # Prepare features
    features = [positions_x, positions_y, positions_z]
    if add_layer_feature:
        features.append(valid_layer_ids)
    
    # Create output arrays
    num_features = len(features)
    locations = np.full((*pixel_ids.shape, num_features), padding_value, dtype=np.float64)
    locations.reshape(-1, num_features)[valid_indices] = np.column_stack(features)
    
    # Handle track IDs
    result_track_ids = np.full(pixel_ids.shape, padding_value, dtype=np.int64)
    if mc_hit_ids is not None and track_ids is not None:
        valid_mc_hits = mc_hit_ids.flatten()[valid_indices]
        track_id_values = track_ids.loc[valid_mc_hits].to_numpy()
        result_track_ids.flat[valid_indices] = track_id_values
    
    return locations, result_track_ids


def convert_mppc_to_locations(mppc_ids: np.ndarray,
                            col_indices: np.ndarray,
                            mppc_positions: pd.DataFrame,
                            mc_hit_ids: Optional[np.ndarray] = None,
                            track_ids: Optional[pd.Series] = None,
                            padding_value: float = DEFAULT_PADDING_VALUE,
                            add_layer_feature: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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
    validate_dataframe_columns(mppc_positions, MPPC_POSITION_COLUMNS, "mppc_positions")
    validate_optional_arrays(mc_hit_ids, track_ids, mppc_ids, ("mc_hit_ids", "track_ids"))
    
    # Setup
    mppc_positions_indexed = mppc_positions.set_index("mppc")
    valid_mask = mppc_ids != padding_value
    valid_indices = valid_mask.flatten()
    
    # Extract valid data
    valid_mppc_ids = mppc_ids.flatten()[valid_indices]
    valid_col_indices = col_indices.flatten()[valid_indices]
    
    # Lookup MPPC data
    try:
        mppc_data = mppc_positions_indexed.loc[valid_mppc_ids]
    except KeyError as e:
        raise DataProcessingError(f"Some MPPC IDs not found in mppc_positions: {e}")
    
    # Calculate 3D positions
    col_offset = MPPC_CENTER_OFFSET - valid_col_indices + PIXEL_OFFSET
    
    positions_x = mppc_data["vx"].values + col_offset * mppc_data["colx"].values
    positions_y = mppc_data["vy"].values + col_offset * mppc_data["coly"].values  
    positions_z = mppc_data["vz"].values + col_offset * mppc_data["colz"].values
    
    # Prepare features
    features = [positions_x, positions_y, positions_z]
    if add_layer_feature:
        layer_values = np.full_like(positions_x, MPPC_LAYER_ID, dtype=np.float32)
        features.append(layer_values)
    
    # Create output arrays
    num_features = len(features)
    locations = np.full((*mppc_ids.shape, num_features), padding_value, dtype=np.float64)
    locations.reshape(-1, num_features)[valid_indices] = np.column_stack(features)
    
    # Handle track IDs
    result_track_ids = np.full(mppc_ids.shape, padding_value, dtype=np.int64)
    if mc_hit_ids is not None and track_ids is not None:
        valid_mc_hits = mc_hit_ids.flatten()[valid_indices]
        track_id_values = track_ids.loc[valid_mc_hits].to_numpy()
        result_track_ids.flat[valid_indices] = track_id_values
    
    return locations, result_track_ids


# ============================================================================
# DATA REMAPPING UTILITIES
# ============================================================================

def remap_track_ids_to_zero_indexed(pixel_track_ids: np.ndarray,
                                   mppc_track_ids: np.ndarray,
                                   padding_value: int = DEFAULT_PADDING_VALUE) -> Tuple[np.ndarray, np.ndarray]:
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
    
    n_events = pixel_track_ids.shape[0]
    
    # Create validity masks
    pixel_valid_mask = (pixel_track_ids != -1) & (pixel_track_ids != padding_value)
    mppc_valid_mask = (mppc_track_ids != -1) & (mppc_track_ids != padding_value)
    pixel_fake_hit_mask = pixel_track_ids == -1
    mppc_fake_hit_mask = mppc_track_ids == -1
    
    # Find minimum track ID for each event
    maximum_track_id = max(np.max(pixel_track_ids), np.max(mppc_track_ids)) + 1
    
    pixel_mins = np.full(n_events, maximum_track_id, dtype=np.int64)
    mppc_mins = np.full(n_events, maximum_track_id, dtype=np.int64)
    
    for i in range(n_events):
        if pixel_valid_mask[i].any():
            pixel_mins[i] = pixel_track_ids[i, pixel_valid_mask[i]].min()
        if mppc_valid_mask[i].any():
            mppc_mins[i] = mppc_track_ids[i, mppc_valid_mask[i]].min()
    
    # Calculate combined minimum for each event
    combined_mins = np.minimum(pixel_mins, mppc_mins)
    combined_mins[combined_mins == maximum_track_id] = 0
    
    # Apply remapping
    pixel_result = pixel_track_ids.astype(np.float32)
    mppc_result = mppc_track_ids.astype(np.float32)
    
    for i in range(n_events):
        if combined_mins[i] != maximum_track_id:
            pixel_result[i, pixel_valid_mask[i]] -= combined_mins[i]
            mppc_result[i, mppc_valid_mask[i]] -= combined_mins[i]
        pixel_result[i, pixel_fake_hit_mask[i]] = -1
        mppc_result[i, mppc_fake_hit_mask[i]] = -1
    
    return pixel_result, mppc_result


def remap_hit_times_to_zero_starting(pixel_hit_times: np.ndarray,
                                    mppc_hit_times: np.ndarray,
                                    padding_value: int = DEFAULT_PADDING_VALUE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remap hit times to start from 0 for each event.
    
    Args:
        pixel_hit_times: Pixel hit time array
        mppc_hit_times: MPPC hit time array
        padding_value: Value used for padding
        
    Returns:
        Tuple of remapped (pixel_hit_times, mppc_hit_times)
    """
    validate_array_shapes(pixel_hit_times, mppc_hit_times)
    
    n_events = pixel_hit_times.shape[0]
    
    # Create validity masks
    pixel_valid_mask = pixel_hit_times != padding_value
    mppc_valid_mask = mppc_hit_times != padding_value
    
    # Find minimum hit time for each event
    pixel_mins = np.full(n_events, np.inf, dtype=pixel_hit_times.dtype)
    mppc_mins = np.full(n_events, np.inf, dtype=mppc_hit_times.dtype)
    
    for i in range(n_events):
        if pixel_valid_mask[i].any():
            pixel_mins[i] = pixel_hit_times[i, pixel_valid_mask[i]].min()
        if mppc_valid_mask[i].any():
            mppc_mins[i] = mppc_hit_times[i, mppc_valid_mask[i]].min()
    
    # Use minimum across both detector types
    combined_mins = np.minimum(pixel_mins, mppc_mins)
    combined_mins[np.isinf(combined_mins)] = 0
    
    # Apply remapping
    pixel_result = pixel_hit_times.copy()
    mppc_result = mppc_hit_times.copy()
    
    for i in range(n_events):
        if not np.isinf(combined_mins[i]):
            pixel_result[i, pixel_valid_mask[i]] -= combined_mins[i]
            mppc_result[i, mppc_valid_mask[i]] -= combined_mins[i]
    
    return pixel_result, mppc_result


# ============================================================================
# TRUTH DATA CREATION
# ============================================================================

def create_track_truth(track_ids_array: np.ndarray,
                      mc_track_truth: pd.DataFrame,
                      data_shape: Tuple[int, ...],
                      padding_value: float = DEFAULT_PADDING_VALUE) -> np.ndarray:
    """Create track truth array from track IDs and MC truth data."""
    track_truth = np.full((*data_shape, 5), padding_value, dtype=np.float32)
    valid_mask = ((track_ids_array != padding_value) & 
                  (track_ids_array != -1)).flatten()
    
    if valid_mask.any():
        valid_track_ids = track_ids_array.flatten()[valid_mask]
        truth_data = mc_track_truth.loc[valid_track_ids].to_numpy(dtype=np.float32)
        track_truth.reshape(-1, 5)[valid_mask] = truth_data
    
    return track_truth


def get_hit_time_truth(mc_hit_ids: np.ndarray,
                      hit_times_series: pd.Series,
                      data_shape: Tuple[int, ...],
                      padding_value: float = DEFAULT_PADDING_VALUE) -> np.ndarray:
    """Create hit time truth array from MC hit IDs."""
    timing_truth = np.full(data_shape, padding_value, dtype=np.float64)
    valid_mask = (mc_hit_ids != padding_value).flatten()
    
    if valid_mask.any():
        valid_mc_hits = mc_hit_ids.flatten()[valid_mask]
        truth_times = hit_times_series.loc[valid_mc_hits].to_numpy()
        timing_truth.flat[valid_mask] = truth_times
    
    return timing_truth


# ============================================================================
# MAIN CONVERSION FUNCTION
# ============================================================================

def convert_root_to_numpy(file_path: Union[str, Path],
                         output_dir: Union[str, Path],
                         output_name: str,
                         padding_value: float = DEFAULT_PADDING_VALUE,
                         hit_cutoff: int = DEFAULT_HIT_CUTOFF,
                         add_layer_feature: bool = False,
                         n_events: Optional[int] = None) -> None:
    """
    Convert ROOT file to NumPy arrays for machine learning.
    
    Args:
        file_path: Path to input ROOT file
        output_dir: Directory to save output files
        output_name: Base name for output files
        padding_value: Value used for padding
        hit_cutoff: Maximum hits per event
        add_layer_feature: Whether to include layer information
        n_events: Maximum number of events to process (optional)
    """
    # Validate inputs
    file_path = validate_file_path(file_path)
    output_dir = Path(output_dir)
    
    if not output_name:
        raise DataProcessingError("Output name must be provided")
    
    if padding_value == -1:
        warnings.warn(
            "Padding value of -1 may conflict with valid data. "
            "Consider using a different padding value.",
            UserWarning,
        )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Loading data from {file_path}...")
        
        # Load data from ROOT file
        with uproot.open(file_path) as file:
            sensor_positions = file["alignment/sensors"].arrays(library="pd")
            mppc_positions = file["alignment/mppcs"].arrays(library="pd")
            fibre_positions = file["alignment/fibres"].arrays(library="pd")
            
            event_kwargs = {"library": "pd"}
            if n_events is not None:
                event_kwargs["entry_stop"] = n_events
            
            event_data = file["mu3e"].arrays(**event_kwargs)
            mc_track_ids = file["mu3e_mchits"]["mc_track"].arrays(library="pd")["mc_track"]
            mc_hit_truth_time = file["mu3e_mchits"]["time"].arrays(library="pd")["time"]
            mc_track_truth = file["mu3e_mc_tracks"].arrays(library="pd")
        
        # Process MC track IDs
        valid_mc_mask = mc_track_ids != -1
        mc_track_ids.loc[valid_mc_mask] = (
            mc_track_truth["mother"].loc[mc_track_ids.loc[valid_mc_mask]].values
        )
        
        if mc_track_ids.isnull().any():
            raise DataProcessingError(
                "Some mc_track_ids do not have corresponding entries in mc_track_truth"
            )
        
        # Validate loaded data
        datasets = [sensor_positions, mppc_positions, fibre_positions, event_data]
        if any(df.empty for df in datasets):
            raise DataProcessingError("One or more required data tables are empty")
        
        # Select relevant columns from MC truth
        mc_track_truth = mc_track_truth[MC_TRACK_TRUTH_COLUMNS]
        
        print("Converting awkward arrays to numpy...")
        
        # Convert awkward arrays to numpy
        series_list = [
            event_data["hit_pixelid"],
            event_data["hit_timestamp"],
            event_data["fibremppc_mppc"],
            event_data["fibremppc_col"],
            event_data["fibremppc_timestamp"],
            event_data["hit_mc_i"],
            event_data["fibremppc_mc_i"],
        ]
        
        (pixel_ids, pixel_timestamps, mppc_ids, mppc_cols, mppc_timestamp,
         pixel_mc_hits, mppc_mc_hits) = load_awkward_to_numpy(
            series_list, max_length=hit_cutoff, fill_value=padding_value
        )
        
        print("Converting IDs to spatial positions...")
        
        # Convert IDs to spatial positions
        pixel_locations, pixel_track_ids = convert_pixels_to_locations(
            pixel_ids, sensor_positions, pixel_mc_hits, mc_track_ids,
            padding_value, add_layer_feature=add_layer_feature
        )
        
        mppc_locations, mppc_track_ids = convert_mppc_to_locations(
            mppc_ids, mppc_cols, mppc_positions, mppc_mc_hits, mc_track_ids,
            padding_value, add_layer_feature=add_layer_feature
        )
        
        # Clean up intermediate data
        del mppc_cols, sensor_positions, mppc_positions, fibre_positions, event_data
        
        print("Converting timestamps to nanoseconds...")
        
        # Convert timestamps to nanoseconds
        pixel_timestamps = convert_pixel_timestamp_to_ns(pixel_timestamps, padding_value)
        mppc_timestamp = convert_mppc_timestamp_to_ns(mppc_timestamp, padding_value)
        
        # Mask timestamps for invalid positions
        pixel_invalid_mask = np.all(pixel_locations == padding_value, axis=-1)
        mppc_invalid_mask = np.all(mppc_locations == padding_value, axis=-1)
        
        pixel_timestamps[pixel_invalid_mask] = padding_value
        mppc_timestamp[mppc_invalid_mask] = padding_value
        pixel_mc_hits[pixel_invalid_mask] = padding_value
        mppc_mc_hits[mppc_invalid_mask] = padding_value
        
        print("Creating truth arrays...")
        
        # Create truth arrays
        pixel_track_truth = create_track_truth(pixel_track_ids, mc_track_truth, pixel_ids.shape)
        mppc_track_truth = create_track_truth(mppc_track_ids, mc_track_truth, mppc_ids.shape)
        
        pixel_timing_truth = get_hit_time_truth(pixel_mc_hits, mc_hit_truth_time, 
                                               pixel_mc_hits.shape, padding_value)
        mppc_timing_truth = get_hit_time_truth(mppc_mc_hits, mc_hit_truth_time,
                                              mppc_mc_hits.shape, padding_value)
        
        print("Remapping track IDs and hit times...")
        
        # Remap track IDs and hit times to start from 0
        pixel_track_ids, mppc_track_ids = remap_track_ids_to_zero_indexed(
            pixel_track_ids, mppc_track_ids, padding_value
        )
        
        pixel_timing_truth, mppc_timing_truth = remap_hit_times_to_zero_starting(
            pixel_timing_truth, mppc_timing_truth, padding_value
        )
        
        # Clean up MC hits arrays
        del pixel_mc_hits, mppc_mc_hits
        
        print("Creating spacetime arrays...")
        
        # Create spacetime arrays
        pixel_spacetime = np.concatenate(
            [pixel_locations, pixel_timestamps[..., None]], axis=-1
        )
        mppc_spacetime = np.concatenate(
            [mppc_locations, mppc_timestamp[..., None]], axis=-1
        )
        
        # Clean up individual arrays
        del pixel_locations, mppc_locations, pixel_timestamps, mppc_timestamp
        
        # Combine track labels
        pixel_track_labels = np.concatenate([
            pixel_track_ids[..., None],
            pixel_timing_truth[..., None],
            pixel_track_truth,
        ], axis=-1)
        
        mppc_track_labels = np.concatenate([
            mppc_track_ids[..., None],
            mppc_timing_truth[..., None],
            mppc_track_truth,
        ], axis=-1)
        
        # Clean up intermediate arrays
        del (pixel_track_ids, mppc_track_ids, pixel_track_truth, 
             mppc_track_truth, pixel_timing_truth, mppc_timing_truth)
        
        print("Filtering events...")
        
        # Filter events with valid hits
        pixel_hit_counts = (pixel_spacetime != padding_value).any(axis=-1).sum(axis=-1)
        mppc_hit_counts = (mppc_spacetime != padding_value).any(axis=-1).sum(axis=-1)
        valid_events = (pixel_hit_counts > 0) & (mppc_hit_counts > 0)
        
        # Apply filter
        pixel_spacetime = pixel_spacetime[valid_events]
        mppc_spacetime = mppc_spacetime[valid_events]
        pixel_track_labels = pixel_track_labels[valid_events]
        mppc_track_labels = mppc_track_labels[valid_events]
        
        print("Saving results...")
        
        # Save results
        file_paths = {
            "pixel_spacetime": output_dir / f"{output_name}_pixel_spacetime.npy",
            "mppc_spacetime": output_dir / f"{output_name}_mppc_spacetime.npy",
            "pixel_labels": output_dir / f"{output_name}_pixel_track_labels.npy",
            "mppc_labels": output_dir / f"{output_name}_mppc_track_labels.npy",
        }
        
        np.save(file_paths["pixel_spacetime"], pixel_spacetime)
        np.save(file_paths["mppc_spacetime"], mppc_spacetime)
        np.save(file_paths["pixel_labels"], pixel_track_labels)
        np.save(file_paths["mppc_labels"], mppc_track_labels)
        
        # Print summary
        print(f"\nSuccessfully processed {file_path}")
        print(f"Events processed: {len(pixel_spacetime)}")
        print(f"Pixel spacetime: {pixel_spacetime.shape} -> {file_paths['pixel_spacetime']}")
        print(f"MPPC spacetime: {mppc_spacetime.shape} -> {file_paths['mppc_spacetime']}")
        print(f"Pixel labels: {pixel_track_labels.shape} -> {file_paths['pixel_labels']}")
        print(f"MPPC labels: {mppc_track_labels.shape} -> {file_paths['mppc_labels']}")
        
    except Exception as e:
        raise DataProcessingError(f"Failed to process ROOT file {file_path}: {str(e)}") from e


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ROOT files to NumPy arrays")
    parser.add_argument("input_file", help="Input ROOT file path")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("output_name", help="Base name for output files")
    parser.add_argument("--padding-value", type=float, default=DEFAULT_PADDING_VALUE,
                       help="Padding value for invalid entries")
    parser.add_argument("--hit-cutoff", type=int, default=DEFAULT_HIT_CUTOFF,
                       help="Maximum hits per event")
    parser.add_argument("--add-layer-feature", action="store_true",
                       help="Add layer information as feature")
    parser.add_argument("--n-events", type=int, help="Maximum number of events to process")
    
    args = parser.parse_args()
    
    convert_root_to_numpy(
        file_path=args.input_file,
        output_dir=args.output_dir,
        output_name=args.output_name,
        padding_value=args.padding_value,
        hit_cutoff=args.hit_cutoff,
        add_layer_feature=args.add_layer_feature,
        n_events=args.n_events
    )