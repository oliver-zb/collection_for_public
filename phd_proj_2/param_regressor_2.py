"""
Parameter Regression using Autoencoder

This program implements a K-fold cross-validated autoencoder with an additional condition on the
latent space, suitable for parameter regression on J-V curves from perovskite solar cell simulations.
The trained model can then be used to predict the underlying parameters from new J-V curve data,
simulated or experimentally measured, providing insights into the physical properties of the solar cells.

Of course, the data set is much smaller than what was actually used in the paper, the number of epochs
would be much higher as well.
"""

from keras.layers import Input, Dense, Conv1D, LeakyReLU, Flatten, Reshape, Conv1DTranspose
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import backend as K
from pathlib import Path
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)

#Hide GPU from visible devices (-1) or choose which GPU to use (0,1) on warp05
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# param_names = ['Top_interface_SRH_rec_vel', 'Pero_carrier_mu', 'Pero_carrier_simplified_tau', 'TiO2_carrier_mu',
#       'Pero_cation_mu', 'Pero_ion_dens']
param_names = ['$v_{rec, SRH}$', '$\mu_{pero}$', r'$\tau_{rec, pero}$', r'$\mu_{TiO}$', r'$\mu_{C}$', r'$\rho_{Ion}$']

DATA_DIR = Path('data')
OUTPUT_DIR = Path('kfold')

EPOCHS = 5
BATCH_SIZE = 128
K_SPLITS = 5
TEST_SIZE = 0.2
PATIENCE = 20

N_SAMPLES = None  # Set to None to load all samples (500), or specify a number smaller than taht

K.clear_session()

def load_data(jv_path, params_path, n_samples = None):
    """
    Load J-V curves and parameters from numpy files.

    Args:
        jv_path: Path to J-V curves file
        params_path: Path to parameters file
        n_samples: Number of samples to load (None for all)

    Returns:
        Tuple of (jv_curves, parameters)
    """
    jv_curves = np.load(jv_path, allow_pickle=True)
    parameters = np.load(params_path, allow_pickle=True)

    if n_samples:
        jv_curves = jv_curves[:n_samples]
        parameters = parameters[:n_samples]

    return jv_curves, parameters

def build_encoder(input_dim, latent_dim):
    """
    Build encoder network.

    Args:
        input_dim: Dimension of input data, defined by the number of simulated points along the J-V curve
        latent_dim: Dimension of latent space, defined by the number of swept parameters

    Returns:
        Keras Model representing the encoder
    """
    x_input = Input(shape=(input_dim,))

    # Reshape for Conv1D
    x = Reshape((input_dim, 1))(x_input)

    # Convolutional layers (probably not really necessary, requires a lot of time in training
    x = Conv1D(128, kernel_size=7, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(64, kernel_size=5, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)

    # Dense layers
    x = Dense(512, activation='relu', name='en_dense_1')(x)
    x = Dense(64, activation='relu', name='en_dense_2')(x)
    x = Dense(32, activation='relu', name='en_dense_3')(x)

    # Bottleneck
    encoded = Dense(latent_dim, activation='relu', name='bottleneck')(x)

    return Model(x_input, encoded, name='encoder')

def build_decoder(latent_dim, output_dim):
    """
    Build decoder network.

    Args:
        latent_dim: Dimension of latent space (same as encoder bottleneck)
        output_dim: Dimension of output data (same as encoder input)
        conv_shape: Shape for convolutional layers

    Returns:
        Keras Model representing the decoder
    """
    encoded_input = Input(shape=(latent_dim,))

    # Dense layers
    x = Dense(32, activation='relu', name='de_dense_1')(encoded_input)
    x = Dense(64, activation='relu', name='de_dense_2')(x)
    x = Dense(512, activation='relu', name='de_dense_3')(x)
    x = Dense(output_dim * 64, activation='relu')(x)

    # Reshape for Conv1DTranspose
    x = Reshape((output_dim, 64))(x)

    # Transposed convolutional layers
    x = Conv1DTranspose(128, kernel_size=5, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1DTranspose(1, kernel_size=7, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    decoded = Flatten()(x)

    return Model(encoded_input, decoded, name='decoder')

def create_autoencoder(input_dim, target_dim):
    """
    Create autoencoder model with encoder and decoder.

    Returns:
        Tuple of (autoencoder, encoder, decoder)
    """
    encoder = build_encoder(input_dim, target_dim)
    decoder = build_decoder(target_dim, input_dim)

    # Create combined autoencoder
    x_input = Input(shape=(input_dim,))
    encoded = encoder(x_input)
    decoded = decoder(encoded)

    autoencoder = Model(x_input, [decoded, encoded])
    autoencoder.compile(
        optimizer='adam',
        loss='mae',
        metrics=['mae'],
        # equal weights for reconstruction and latent losses, can be tuned if desired
        loss_weights=[1, 1]
    )

    return autoencoder, encoder, decoder

def calculate_metrics(y_true, y_pred):
    """
    Calculate R² score and RMSE.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Tuple of (r2_score, rmse)
    """
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    return r2, rmse


def train_fold(X_train, X_val, y_train, y_val, split_idx, k_splits):
    """
    Train model on one fold.

    Args:
        X_train: Training data J-V curves
        X_val: Validation data J-V curves
        y_train: Training data varied parameters
        y_val: Validation data varied parameters
        split_idx: Current fold index
        k_splits: Total number of folds

    Returns:
        Tuple of (trained_model, history_dict of this model)
    """
    print(f'\n--- Fold {split_idx + 1}/{k_splits} ---')

    input_dim = X_train.shape[1]
    target_dim = y_train.shape[1]

    ae, encoder, decoder = create_autoencoder(input_dim, target_dim)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )

    history = ae.fit(
        [X_train], [X_train, y_train],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=[[X_val], [X_val, y_val]],
        callbacks=[early_stopping],
        verbose=1
    )

    return ae, history.history


def evaluate_on_test_set(model, X_test, y_test):
    """
    Evaluate model on test data.

    Returns:
        Dictionary of evaluation metrics
    """
    predictions = model.predict(X_test)
    x_pred, y_pred = predictions[0], predictions[1]

    r2_rec, rmse_rec = calculate_metrics(X_test, x_pred)
    r2_lat, rmse_lat = calculate_metrics(y_test, y_pred)

    metrics = {
        'r2_reconstruction': r2_rec,
        'r2_latent': r2_lat,
        'rmse_reconstruction': rmse_rec,
        'rmse_latent': rmse_lat
    }

    print(f"R² (reconstruction): {r2_rec:.8f}")
    print(f"R² (latent): {r2_lat:.8f}")
    print(f"RMSE (reconstruction): {rmse_rec:.8f}")
    print(f"RMSE (latent): {rmse_lat:.8f}")

    return metrics


def run_cross_validation(X, y):
    """
    Run K-fold cross-validation.

    Args:
        X: Input features (J-V data)
        y: Target values (device parameters)

    Returns:
        Dictionary containing all metrics across folds
    """
    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=False
    )

    # Initialize metrics storage
    metrics_storage = {
        'train_losses': [],
        'val_losses': [],
        'decoder_train_mae': [],
        'decoder_val_mae': [],
        'latent_train_mae': [],
        'latent_val_mae': [],
        'r2_reconstruction': [],
        'r2_latent': [],
        'rmse_reconstruction': [],
        'rmse_latent': []
    }

    best_val_loss = float('inf')

    # K-fold cross-validation
    kf = KFold(n_splits=K_SPLITS)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
        # Split fold data
        X_train = X_trainval[train_idx]
        X_val = X_trainval[val_idx]
        y_train = y_trainval[train_idx]
        y_val = y_trainval[val_idx]

        # Train fold
        ae, history = train_fold(X_train, X_val, y_train, y_val, fold_idx, K_SPLITS)

        # Store training history
        metrics_storage['train_losses'].append(np.array(history['loss']))
        metrics_storage['val_losses'].append(np.array(history['val_loss']))
        metrics_storage['decoder_train_mae'].append(np.array(history['decoder_mae']))
        metrics_storage['decoder_val_mae'].append(np.array(history['val_decoder_mae']))
        metrics_storage['latent_train_mae'].append(np.array(history['encoder_mae']))
        metrics_storage['latent_val_mae'].append(np.array(history['val_encoder_mae']))

        # Evaluate on test set
        test_metrics = evaluate_on_test_set(ae, X_test, y_test)

        metrics_storage['r2_reconstruction'].append(test_metrics['r2_reconstruction'])
        metrics_storage['r2_latent'].append(test_metrics['r2_latent'])
        metrics_storage['rmse_reconstruction'].append(test_metrics['rmse_reconstruction'])
        metrics_storage['rmse_latent'].append(test_metrics['rmse_latent'])

        # Update best model
        final_val_loss = history['val_loss'][-1]
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            print(f'New best validation loss: {final_val_loss:.8f}')

    print(f'\n=== Cross-Validation Complete ===')
    print(f'Best validation loss: {best_val_loss:.8f}')

    return metrics_storage


def main():
    """Main execution function."""
    # Clear Keras session
    K.clear_session()

    # Load data
    jv_curves, parameters = load_data(
        DATA_DIR / 'jvcurves_short.npy',
        DATA_DIR / 'params_short.npy',
        n_samples=N_SAMPLES
    )

    # Run cross-validation
    metrics = run_cross_validation(jv_curves, parameters)

    # Optionally save results
    # OUTPUT_DIR.mkdir(exist_ok=True)
    # np.save(OUTPUT_DIR / 'metrics.npy', metrics)


if __name__ == '__main__':
    main()