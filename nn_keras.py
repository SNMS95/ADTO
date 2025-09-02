from functools import partial
import numpy as np
import os
# Check that Keras backend has already been set
# before importing keras
assert os.environ.get("KERAS_BACKEND") is not None
import keras  # noqa


def get_optimizer(opt_str, **hyper_params):
    if opt_str == "adam":
        return keras.optimizers.Adam(**hyper_params)
    elif opt_str == "sgd":
        return keras.optimizers.SGD(**hyper_params)
    elif opt_str == "rmsprop":
        return keras.optimizers.RMSprop(**hyper_params)
    elif opt_str == "adagrad":
        return keras.optimizers.Adagrad(**hyper_params)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_str}")


def nn_input(nn_type, Nx=64, Ny=64, latent_size=128):
    """Generate required neural network inputs"""
    if nn_type in ["mlp", "siren"]:
        # Need coordinates of element centroids
        x_centers = np.linspace(-1 + 1/Nx, 1 - 1/Nx, Nx)
        y_centers = np.linspace(-1 + 1/Ny, 1 - 1/Ny, Ny)
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='xy')
        # Stack coordinates with Fortran-style ordering
        input_to_net = np.column_stack(
            [x_grid.ravel(order='F'), y_grid.ravel(order='F')])
        return input_to_net
    elif nn_type == "cnn":
        return np.random.normal(size=(latent_size,)).reshape(1, latent_size)
    else:
        raise ValueError(f"Unsupported nn_type: {nn_type}")


def create_network_and_input(nn_type: str = "mlp", hyper_params: dict = None,
                             random_seed: int = 42, grid_size: tuple = (32, 64)):
    """Create neural topology optimization model."""
    keras.backend.clear_session()  # We need this to prevent memory overload
    keras.utils.set_random_seed(random_seed)

    if hyper_params is None:
        hyper_params = {}

    Ny, Nx = grid_size
    latent_size = hyper_params.get("latent_size", 128)

    if nn_type == "mlp":
        n_h_layers = hyper_params.get("num_hidden_layers", 5)
        units = hyper_params.get("hidden_units", 20)

        inputs = keras.layers.Input(shape=(2,), name='coordinates')
        x = inputs
        for i in range(n_h_layers):
            x = keras.layers.Dense(units)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Dense(1)(x)

    elif nn_type == "cnn":
        latent_size = hyper_params.get("latent_size", 128)
        activation = hyper_params.get("activation", "tanh")

        inputs = keras.layers.Input(shape=(latent_size,))
        filters = (Ny//8) * (Nx//8) * 32
        x = keras.layers.Dense(
            filters,
            kernel_initializer=keras.initializers.Orthogonal())(inputs)
        x = keras.layers.Reshape([Ny//8, Nx//8, 32])(x)

        for resize, nf in zip([1, 2, 2, 2, 1], [64, 32, 16, 8, 1]):
            x = keras.layers.Activation(activation)(x)
            x = keras.layers.UpSampling2D(
                (resize, resize), interpolation='bilinear')(x)
            x = keras.layers.LayerNormalization()(x)
            x = keras.layers.Conv2D(nf, 5, padding="same")(x)
        outputs = keras.layers.Reshape([Ny, Nx])(keras.layers.Flatten()(x))

    elif nn_type == "siren":
        omega0 = hyper_params.get("frequency_factor", 30.0)
        layers = hyper_params.get("num_hidden_layers", 3)
        units = hyper_params.get("hidden_units", 256)

        def sine_init(shape, dtype=None, first=False):
            limit = 1/shape[0] if first else (6/shape[0])**0.5/omega0
            return keras.random.uniform(shape, -limit, limit, seed=random_seed)

        inputs = keras.layers.Input(shape=(2,))
        x = keras.layers.Dense(units, kernel_initializer=partial(
            sine_init, first=True))(inputs)
        x = keras.ops.sin(x * omega0)
        for _ in range(layers-1):
            x = keras.ops.sin(keras.layers.Dense(
                units, kernel_initializer=sine_init)(x) * omega0)
        outputs = keras.layers.Dense(1, kernel_initializer=sine_init)(x)

    else:
        raise ValueError(f"Unsupported nn_type: {nn_type}")

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name=f'{nn_type.upper()}')
    input_to_net = nn_input(nn_type, Nx=Nx, Ny=Ny, latent_size=latent_size)
    # Build model to populate the weights and biases
    _ = model(input_to_net)
    return model, input_to_net
