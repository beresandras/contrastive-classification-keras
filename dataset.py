import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_dataset(steps_per_epoch):
    # labeled and unlabeled samples are loaded synchronously
    # with batch sizes selected accordingly
    unlabeled_batch_size = 100000 // steps_per_epoch
    labeled_batch_size = 5000 // steps_per_epoch
    batch_size = unlabeled_batch_size + labeled_batch_size
    print(
        "batch size is {} (unlabeled) + {} (labeled)".format(
            unlabeled_batch_size, labeled_batch_size
        )
    )

    unlabeled_train_dataset = (
        tfds.load("stl10", split="unlabelled", as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=5000)
        .batch(unlabeled_batch_size, drop_remainder=True)
    )
    labeled_train_dataset = (
        tfds.load("stl10", split="train", as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=5000)
        .batch(labeled_batch_size, drop_remainder=True)
    )
    test_dataset = (
        tfds.load("stl10", split="test", as_supervised=True)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    # labeled and unlabeled datasets are zipped together
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return batch_size, train_dataset, labeled_train_dataset, test_dataset