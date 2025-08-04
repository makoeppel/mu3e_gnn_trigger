import numpy as np


def cartesian_to_cylindrical(data, padding_value=-1):
    """Convert Cartesian coordinates to cylindrical coordinates.
    Args:
        data (np.ndarray): Input data in Cartesian coordinates with shape (N, M, 3) or .
        padding_value (int, optional): Value to use for padding. Defaults to -1.
    Returns:
        np.ndarray: Data in cylindrical coordinates with shape (N, M, 3).
    """
    if data.ndim != 3:
        raise ValueError("Input data must have shape (N, M, 3) for Cartesian coordinates.")
    if data.shape[-1] != 3 and data.shape[-1] != 2:
        raise ValueError("Input data must have 3 features for Cartesian coordinates (x, y, z).") 
    r = np.sqrt(data[:, :, 0]**2 + data[:, :, 1]**2)
    theta = np.arctan2(data[:, :, 1], data[:, :, 0])
    if data.shape[-1] == 3:
        z = data[:, :, 2]
        cylindrical_data = np.stack((r, theta, z), axis=-1)
    else:
        cylindrical_data = np.stack((r, theta), axis=-1)
    cylindrical_data[data[:, :, 0] == padding_value] = padding_value
    return cylindrical_data

def normalize_data(data, type : str = "minmax", feature_axis = -1, feature_range: tuple = (0, 1), padding_value = -1):
    data = np.moveaxis(data, feature_axis, -1)

    if type == "minmax":
        mask = (data != padding_value).any(axis=-1)
        min_vals = np.min(np.where(mask[..., None], data, np.inf), axis=(0, 1))
        max_vals = np.max(np.where(mask[..., None], data, -np.inf), axis=(0, 1))
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = np.nan  # Avoid division by zero

        normalized_data = (data - min_vals) / range_vals
        scaled_data = normalized_data * (feature_range[1] - feature_range[0]) + feature_range[0]
        scaled_data[~mask] = padding_value
        scaled_data = np.nan_to_num(scaled_data, nan=padding_value)

        data = np.moveaxis(scaled_data, -1, feature_axis)
        return data
    
    elif type == "zscore":
        mask = (data != padding_value).any(axis=-1)
        mean_vals = np.mean(np.where(mask[..., None], data, 0), axis=(0, 1))
        std_vals = np.std(np.where(mask[..., None], data, 0), axis=(0, 1))
        std_vals[std_vals == 0] = np.nan
        normalized_data = (data - mean_vals) / std_vals
        normalized_data[~mask] = padding_value
        normalized_data = np.nan_to_num(normalized_data, nan=padding_value)
        data = np.moveaxis(normalized_data, -1, feature_axis)
        return data
    else:
        raise ValueError("Unsupported normalization type. Use 'minmax' or 'zscore'.")
    

def change_padding_value(data, padding_value, new_padding_value):
    data[data == padding_value] = new_padding_value
    return data