import numpy as np
from plotting import plot_raw_data_xy, plot_multiple_events_xy, plot_event
from autoencoder import create_vae

def create_cnn_matrix_dataset(bkg_arr, sig_arr, mix_fraction=0.5, max_hits=256, seed=42, use_time=True, min_hits=4):
    np.random.seed(seed)
    num_bkg_events = bkg_arr.shape[0]
    num_sig_events = sig_arr.shape[0]

    matrices = []
    labels = []

    mix_mask = np.random.rand(num_bkg_events) < mix_fraction
    sig_indices_pool = list(np.random.permutation(num_sig_events))
    mixing = True

    for i in range(num_bkg_events):
        bkg_event = bkg_arr[i]
        valid_bkg = bkg_event[bkg_event[:, 0] != -999.0]
        if not use_time: valid_bkg = valid_bkg[:,:3]

        if mix_mask[i] and mixing:
            if len(sig_indices_pool) == 0:
                print(f"Warning: Out of unique sig events at bkg index {i}. Stopping mix allocation.")
                mixing = False
                continue

            rand_cos_idx = sig_indices_pool.pop()

            sig_event = sig_arr[rand_cos_idx]
            valid_sig = sig_event[sig_event[:, 0] != -999.0]

            if not use_time: valid_sig = valid_sig[:,:3]

            if len(valid_sig) >= min_hits:

                combined_hits = np.random.permutation(
                    np.vstack([valid_bkg, valid_sig])
                )
                label = 1.0  
            else:
                combined_hits = np.random.permutation(valid_bkg)
                label = 0.0
        else:
            combined_hits = np.random.permutation(valid_bkg)
            label = 0.0

        if len(combined_hits) == 0:
            continue

        if len(combined_hits) > max_hits:
            combined_hits = combined_hits[:max_hits]

        if use_time: event_matrix = np.zeros((1, max_hits, 4))
        if not use_time: event_matrix = np.zeros((1, max_hits, 3))

        num_actual_hits = combined_hits.shape[0]
        # Assign hits directly into the width timeline
        event_matrix[0, :num_actual_hits, :] = combined_hits

        matrices.append(event_matrix)
        labels.append(label)

    X = np.array(matrices)
    y = np.array(labels)

    return X, y

def normalize(x):
    x = np.asarray(x, dtype=np.float32)

    x_min = x.min()
    x_max = x.max()

    # Prevent division by zero
    if x_max == x_min:
        return np.zeros_like(x), x_min, x_max

    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm, x_min, x_max


def load_data(path_bkg, path_sig, plotting=False, max_hits=256, use_autoencoder=False, use_time=True):
    sig_data = np.load(path_sig)
    bkg_data = np.load(path_bkg)

    if use_autoencoder:
        ids_sig = sig_data[:, :, 0].flatten()
        ids_bkg = bkg_data[:, :, 0].flatten()
        ids = np.concatenate([ids_sig, ids_bkg])
        ids = ids[ids != -999]
        ids, _, _ = normalize(ids)

        vae_model, latent_model = create_vae(encoding_dim = 3, layers=[8, 16, 32])
        vae_model.fit(
            ids, ids,
            epochs=1,
            batch_size=32,
            verbose=1
        )
        z_mean_sig = latent_model(normalize(ids_sig))
        z_mean_sig.reshape(ids_sig.shape[0], ids_sig.shape[1], 3)
        print(z_mean_sig, z_mean_sig.shape)

    if plotting:
        plot_raw_data_xy(bkg_data, sig_data)

    X, y = create_cnn_matrix_dataset(bkg_data, sig_data, mix_fraction=0.5, max_hits=max_hits, use_time=use_time)

    X_beam = X[y==0]
    X_cosmic = X[y==1]
    len_beam = []
    len_cosmic = []
    for xi in X_beam: len_beam.append(len(xi[xi!=0]))
    for xi in X_cosmic: len_cosmic.append(len(xi[xi!=0]))
    print(np.mean(len_beam), np.mean(len_cosmic))

    plot_event(X[y==0][0].T, name="example_beam_only.pdf")
    plot_event(X[y==1][1].T, name="example_beam_with_cosmic.pdf")

    if plotting:
        plot_multiple_events_xy(X, y)

    print(print("Ratio sig/(bkg+sig)", sum(y)/len(y)))

    return X, y
