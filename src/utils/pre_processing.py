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
        raise ValueError(
            "Input data must have shape (N, M, 3) for Cartesian coordinates."
        )
    if data.shape[-1] != 3 and data.shape[-1] != 2:
        raise ValueError(
            "Input data must have 3 features for Cartesian coordinates (x, y, z)."
        )
    r = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2)
    theta = np.arctan2(data[:, :, 1], data[:, :, 0])
    if data.shape[-1] == 3:
        z = data[:, :, 2]
        cylindrical_data = np.stack((r, theta, z), axis=-1)
    else:
        cylindrical_data = np.stack((r, theta), axis=-1)
    cylindrical_data[data[:, :, 0] == padding_value] = padding_value
    return cylindrical_data


def normalize_data(
    data,
    type: str = "minmax",
    feature_axis=-1,
    feature_range: tuple = (0, 1),
    padding_value=-1,
):
    """Normalize data along the specified feature axis.
    Args:
        data (np.ndarray): Input data to normalize.
        type (str, optional): Normalization type, either "minmax" or "zscore". Defaults to "minmax".
        feature_axis (int, optional): Axis along which to normalize the data. Defaults to -1 (last axis).
        feature_range (tuple, optional): Range for min-max normalization. Defaults to (0, 1).
        padding_value (int, optional): Value to use for padding. Defaults to -1.
    Returns:
        np.ndarray: Normalized data.
    """
    data = np.moveaxis(data, feature_axis, -1)

    if type == "minmax":
        mask = (data != padding_value).any(axis=feature_axis)
        min_vals = np.min(np.where(mask[..., None], data, np.inf), axis=(0, 1))
        max_vals = np.max(np.where(mask[..., None], data, -np.inf), axis=(0, 1))
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = np.nan  # Avoid division by zero

        normalized_data = (data - min_vals) / range_vals
        scaled_data = (
            normalized_data * (feature_range[1] - feature_range[0]) + feature_range[0]
        )
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
    """Change the padding value in a numpy array.
    Args:
        data (np.ndarray): Input data.
        padding_value (int): Current padding value to change.
        new_padding_value (int): New padding value to set.
    Returns:
        np.ndarray: Data with updated padding value.
    """
    data[data == padding_value] = new_padding_value
    return data


def reorder_spacetime(spacetime, padding_value=-1):
    """
    Reorder spacetime data so that hits in each event are sorted by time.
    Vectorized version for speed. Assumes time is the last feature in the spacetime array.
    Args:
        spacetime (np.ndarray): Input spacetime data with shape (N_events, N_hits, N_features).
        padding_value (int, optional): Value to use for padding. Defaults to -1.
    Returns:
        reordered (np.ndarray): Reordered spacetime data with the same shape as input.
    """
    if spacetime.ndim != 3:
        raise ValueError(
            "Input spacetime must be a 3D array (N_events, N_hits, N_features)"
        )
    n_events, n_hits, n_features = spacetime.shape

    reordered = np.full_like(spacetime, padding_value)

    valid_mask = (spacetime != padding_value).all(axis=-1)

    times = np.where(valid_mask, spacetime[:, :, -1], np.inf)

    sort_indices = np.argsort(times, axis=1, kind="stable")

    row_idx = np.arange(n_events)[:, None]
    reordered = spacetime[row_idx, sort_indices]

    return reordered


def ContrastSamples(
    bg_pixel_spacetime,
    sig_pixel_spacetime,
    bg_mppc_spacetime,
    sig_mppc_spacetime,
    num_samples=1000,
    padding_value=-1,
):
    """
    Create contrastive samples from background and signal spacetime data. Randomly selects background and signal events,
    ensuring that the total length of pixel and MPPC sequences does not exceed the maximum length. Constructs a negative signal (outlier) event by combining
    a background pixel and MPPC event with a signal pixel and MPPC event, and a positive background event by combining two background pixel and MPPC events.
    Args:
        bg_pixel_spacetime (np.ndarray): Background pixel spacetime data with shape (N_events, N_hits, N_features).
        sig_pixel_spacetime (np.ndarray): Signal pixel spacetime data with shape (N_events, N_hits, N_features).
        bg_mppc_spacetime (np.ndarray): Background MPPC spacetime data with shape (N_events, N_hits, N_features).
        sig_mppc_spacetime (np.ndarray): Signal MPPC spacetime data with shape (N_events, N_hits, N_features).
        num_samples (int, optional): Number of contrastive samples to generate. Defaults to 1000.
        padding_value (int, optional): Value to use for padding. Defaults to -1.

    Returns:
        pure_bg_pixel, pure_bg_mppc,
        contrast_pixel_signal, contrast_mppc_signal,
        contrast_pixel_background, contrast_mppc_background
    """

    max_pixel_length = bg_pixel_spacetime.shape[1]
    max_mppc_length = bg_mppc_spacetime.shape[1]

    pure_bg_pixel = np.full(
        (num_samples, *bg_pixel_spacetime.shape[1:]), padding_value, dtype=np.float32
    )
    pure_bg_mppc = np.full(
        (num_samples, *bg_mppc_spacetime.shape[1:]), padding_value, dtype=np.float32
    )

    contrast_pixel_signal = np.full(
        (num_samples, *sig_pixel_spacetime.shape[1:]), padding_value, dtype=np.float32
    )
    contrast_mppc_signal = np.full(
        (num_samples, *sig_mppc_spacetime.shape[1:]), padding_value, dtype=np.float32
    )

    contrast_pixel_background = np.full(
        (num_samples, *bg_pixel_spacetime.shape[1:]), padding_value, dtype=np.float32
    )
    contrast_mppc_background = np.full(
        (num_samples, *bg_mppc_spacetime.shape[1:]), padding_value, dtype=np.float32
    )

    def valid_mask(arr):
        """Mask for valid (non-padding) rows."""
        return (arr != padding_value).all(axis=-1)

    bg_pixel_masks = valid_mask(bg_pixel_spacetime)
    bg_mppc_masks = valid_mask(bg_mppc_spacetime)
    sig_pixel_masks = valid_mask(sig_pixel_spacetime)
    sig_mppc_masks = valid_mask(sig_mppc_spacetime)

    bg_pixel_lengths = bg_pixel_masks.sum(axis=-1).flatten()
    bg_mppc_lengths = bg_mppc_masks.sum(axis=-1).flatten()
    sig_pixel_lengths = sig_pixel_masks.sum(axis=-1).flatten()
    sig_mppc_lengths = sig_mppc_masks.sum(axis=-1).flatten()

    bg_indices = np.arange(bg_pixel_spacetime.shape[0])
    sig_indices = np.arange(sig_pixel_spacetime.shape[0])

    used_bg_indices = set()
    used_sig_indices = set()
    index_combination = set()

    def copy_sequence(dest, dest_idx, src, mask, offset=0):
        length = mask.sum()
        dest[dest_idx, offset : offset + length] = src[mask]

    selected_samples = 0
    max_iterations = num_samples * 20
    iterations = 0
    hit_diff_tolerance = 5

    while selected_samples < num_samples:
        iterations += 1

        bg_sample = np.random.randint(len(bg_indices))
        sig_sample = np.random.randint(len(sig_indices))

        if (bg_sample, sig_sample) in index_combination:
            continue

        pixel_length = bg_pixel_lengths[bg_sample] + sig_pixel_lengths[sig_sample]
        mppc_length = bg_mppc_lengths[bg_sample] + sig_mppc_lengths[sig_sample]

        if pixel_length > max_pixel_length or mppc_length > max_mppc_length:
            continue

        candidates = np.where(
            (bg_pixel_lengths < max_pixel_length - bg_pixel_lengths[bg_sample]) & (np.abs(bg_pixel_lengths + bg_pixel_lengths[bg_sample] - pixel_length) <= hit_diff_tolerance)
            & (bg_mppc_lengths < max_mppc_length - bg_mppc_lengths[bg_sample]) & (np.abs(bg_mppc_lengths + bg_mppc_lengths[bg_sample] - mppc_length) <= hit_diff_tolerance)
            & (bg_indices != bg_sample)
        )[0]
        if candidates.size == 0:
            continue
        smaller_bg = np.random.choice(candidates)

        copy_sequence(
            pure_bg_pixel,
            selected_samples,
            bg_pixel_spacetime[bg_sample],
            bg_pixel_masks[bg_sample],
        )
        copy_sequence(
            pure_bg_mppc,
            selected_samples,
            bg_mppc_spacetime[bg_sample],
            bg_mppc_masks[bg_sample],
        )

        copy_sequence(
            contrast_pixel_signal,
            selected_samples,
            bg_pixel_spacetime[bg_sample],
            bg_pixel_masks[bg_sample],
        )
        copy_sequence(
            contrast_pixel_signal,
            selected_samples,
            sig_pixel_spacetime[sig_sample],
            sig_pixel_masks[sig_sample],
            offset=bg_pixel_lengths[bg_sample],
        )

        copy_sequence(
            contrast_mppc_signal,
            selected_samples,
            bg_mppc_spacetime[bg_sample],
            bg_mppc_masks[bg_sample],
        )
        copy_sequence(
            contrast_mppc_signal,
            selected_samples,
            sig_mppc_spacetime[sig_sample],
            sig_mppc_masks[sig_sample],
            offset=bg_mppc_lengths[bg_sample],
        )

        copy_sequence(
            contrast_pixel_background,
            selected_samples,
            bg_pixel_spacetime[smaller_bg],
            bg_pixel_masks[smaller_bg],
        )
        copy_sequence(
            contrast_pixel_background,
            selected_samples,
            bg_pixel_spacetime[bg_sample],
            bg_pixel_masks[bg_sample],
            offset=bg_pixel_lengths[smaller_bg],
        )

        copy_sequence(
            contrast_mppc_background,
            selected_samples,
            bg_mppc_spacetime[smaller_bg],
            bg_mppc_masks[smaller_bg],
        )
        copy_sequence(
            contrast_mppc_background,
            selected_samples,
            bg_mppc_spacetime[bg_sample],
            bg_mppc_masks[bg_sample],
            offset=bg_mppc_lengths[smaller_bg],
        )

        used_bg_indices.add(bg_sample)
        used_sig_indices.add(sig_sample)
        index_combination.add((bg_sample, sig_sample))
        selected_samples += 1

    print(
        f"Used {len(used_bg_indices)}/{bg_pixel_spacetime.shape[0]} background events."
    )
    print(f"Used {len(used_sig_indices)}/{sig_pixel_spacetime.shape[0]} signal events.")

    # Reorder spacetime data for contrast samples
    contrast_pixel_signal = reorder_spacetime(contrast_pixel_signal, padding_value)
    contrast_mppc_signal = reorder_spacetime(contrast_mppc_signal, padding_value)
    contrast_pixel_background = reorder_spacetime(
        contrast_pixel_background, padding_value
    )
    contrast_mppc_background = reorder_spacetime(
        contrast_mppc_background, padding_value
    )
    return (
        pure_bg_pixel,
        pure_bg_mppc,
        contrast_pixel_signal,
        contrast_mppc_signal,
        contrast_pixel_background,
        contrast_mppc_background,
    )
