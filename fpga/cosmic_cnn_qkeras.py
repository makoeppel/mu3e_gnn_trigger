#

from sklearn.model_selection import train_test_split
from plotting import plot_efficiency_vs_acceptance, plot_auc_curve
from load_data import load_data
from cnn import create_cnn_model


if __name__ == "__main__":
    TUNABLE_MAX_HITS = 32
    TUNABLE_CONV_CHANNELS = [8,16,32]
    TUNABLE_DENSE_DIM = 32
    TUNABLE_USE_TIME = True
    TUNABLE_USE_AUTOENCODER = False

    print("Setting up tracking datasets...")
    if TUNABLE_USE_AUTOENCODER:
        beam_path = "../data/beam100k_pixel_idtime.npy"
        cosmic_path = "../data/cosmic100k_pixel_idtime.npy"
        sig_path = "../data/sig30k_pixel_idtime.npy"
    else:
        beam_path = "../data/beam100k_pixel_spacetime.npy"
        cosmic_path = "../data/cosmic100k_pixel_spacetime.npy"
        sig_path = "../data/sig30k_pixel_spacetime.npy"

    X, y = load_data(beam_path, cosmic_path, plotting=False, use_time=TUNABLE_USE_TIME, max_hits=TUNABLE_MAX_HITS, use_autoencoder=TUNABLE_USE_AUTOENCODER)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, total_params = create_cnn_model(
        max_hits=TUNABLE_MAX_HITS, 
        num_channels=TUNABLE_CONV_CHANNELS, 
        dense_dim=TUNABLE_DENSE_DIM,
        use_time=TUNABLE_USE_TIME
    )

    print("\nBeginning QKeras Quantization Aware Training Execution loop...")
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    print("\nGenerating final tracking evaluation profile...")
    plot_efficiency_vs_acceptance(model, X_test, y_test, TUNABLE_MAX_HITS, total_params)
    plot_auc_curve(model, X_test, y_test)
