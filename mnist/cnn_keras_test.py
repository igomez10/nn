import logging
from dataclasses import dataclass

import keras
import pytest
from keras import layers


@dataclass
class ModelCase:
    name: str
    model: keras.Model
    optimizer: str
    loss: str
    epochs: int


logger = logging.getLogger(__name__)


def test_model_trains():
    """Each case trains its model and beats random guessing."""
    cases = [
        ModelCase(
            name="cnn-adam-1ep",
            model=keras.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Conv2D(32, 3, padding="same", activation="relu"),
                    layers.Conv2D(32, 3, padding="same", activation="relu"),
                    layers.MaxPooling2D(),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(10, activation="softmax"),
                ]
            ),
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            epochs=10,
        ),
        ModelCase(
            name="cnn-rmsprop-2ep",
            model=keras.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Conv2D(32, 3, padding="same", activation="relu"),
                    layers.Conv2D(32, 3, padding="same", activation="relu"),
                    layers.MaxPooling2D(),
                    layers.Conv2D(64, 3, padding="same", activation="relu"),
                    layers.Conv2D(64, 3, padding="same", activation="relu"),
                    layers.MaxPooling2D(),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(10, activation="softmax"),
                ]
            ),
            optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            epochs=2,
        ),
        ModelCase(
            name="mlp-adam-1ep",
            model=keras.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(10, activation="softmax"),
                ]
            ),
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            epochs=1,
        ),
        ModelCase(
            name="mlp-sgd-3ep",
            model=keras.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(10, activation="softmax"),
                ]
            ),
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
            epochs=3,
        ),
    ]

    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    # split off a validation set from the training data
    x_train, x_val = x_train[:-10000], x_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    x_test, y_test = keras.datasets.mnist.load_data()[1]

    for case in cases:
        case.model.compile(
            optimizer=case.optimizer, loss=case.loss, metrics=["accuracy"]
        )
        history = case.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=case.epochs,
            batch_size=64,
            verbose=0,
        )

        predictions = case.model.predict(x_test[:4], verbose=0)
        assert predictions.shape == (4, 10), case.name
        # Softmax output: each row sums to ~1.
        assert predictions.sum(axis=1) == pytest.approx(1.0, abs=1e-4), case.name

        # Random guessing over 10 classes is 0.1; training should comfortably exceed it.
        assert history.history["accuracy"][-1] > 0.5, case.name

        # print evaluation metrics for manual inspection
        loss, accuracy = case.model.evaluate(x_test, y_test, verbose=0)
        logger.info(
            f"{case.name}: final accuracy={history.history['accuracy'][-1]:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
