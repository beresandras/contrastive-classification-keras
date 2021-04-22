import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from dataset import prepare_dataset
from augmentations import RandomResizedCrop, RandomColorJitter
from algorithms import BarlowTwins, SimCLR, MoCo

tf.get_logger().setLevel("WARN")  # suppress info-level logs

# hyperparameters
num_epochs = 30
steps_per_epoch = 200
width = 128
redundancy_reduction_weight = 10.0
temperature = 0.1
momentum_coeff = 0.99

# hyperparameters corresponding to each algorithm
hyperparams = {
    BarlowTwins: {"redundancy_reduction_weight": redundancy_reduction_weight},
    SimCLR: {"temperature": temperature},
    MoCo: {
        "temperature": temperature,
        "momentum_coeff": momentum_coeff,
    },
}

# load STL10 dataset
batch_size, train_dataset, labeled_train_dataset, test_dataset = prepare_dataset(
    steps_per_epoch
)

# select an algorithm
Algorithm = SimCLR  # BarlowTwins, SimCLR, MoCo

# architecture
model = Algorithm(
    contrastive_augmenter=keras.Sequential(
        [
            layers.Input(shape=(96, 96, 3)),
            preprocessing.Rescaling(1 / 255),
            preprocessing.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        ],
        name="contrastive_augmenter",
    ),
    classification_augmenter=keras.Sequential(
        [
            layers.Input(shape=(96, 96, 3)),
            preprocessing.Rescaling(1 / 255),
            preprocessing.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ],
        name="classification_augmenter",
    ),
    encoder=keras.Sequential(
        [
            layers.Input(shape=(96, 96, 3)),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
        ],
        name="encoder",
    ),
    projection_head=keras.Sequential(
        [
            layers.Input(shape=(width,)),
            layers.Dense(width, activation="relu"),
            layers.Dense(width),
        ],
        name="projection_head",
    ),
    linear_probe=keras.Sequential(
        [
            layers.Input(shape=(width,)),
            layers.Dense(10),
        ],
        name="linear_probe",
    ),
    **hyperparams[Algorithm],
)

# optimizers
model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)

# run training
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

# save history
with open("{}.pkl".format(Algorithm.__name__), "wb") as write_file:
    pickle.dump(history, write_file)
