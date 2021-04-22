import tensorflow as tf

from abc import abstractmethod
from tensorflow import keras


class ContrastiveModel(keras.Model):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
    ):
        super().__init__()

        self.contrastive_augmenter = contrastive_augmenter
        self.classification_augmenter = classification_augmenter
        self.encoder = encoder
        self.projection_head = projection_head
        self.linear_probe = linear_probe

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy()

    def reset_metrics(self):
        self.contrastive_accuracy.reset_states()
        self.probe_accuracy.reset_states()

    def update_contrastive_accuracy(self, projections_1, projections_2):
        # cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = tf.matmul(projections_1, projections_2, transpose_b=True)

        # the similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    @abstractmethod
    def contrastive_loss(self, projections_1, projections_2):
        pass

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        # both labeled and unlabeled images are used, without labels
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        # each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            # the representations are passed through a projection mlp
            projections_1 = self.projection_head(self.encoder(augmented_images_1))
            projections_2 = self.projection_head(self.encoder(augmented_images_2))
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(projections_1, projections_2)

        # labels are only used in evalutation for an on-the-fly logistic regression
        preprocessed_images = self.classification_augmenter(labeled_images)
        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_accuracy.update_state(labels, class_logits)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result(),
        }

    def test_step(self, data):
        labeled_images, labels = data

        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        self.probe_accuracy.update_state(labels, class_logits)
        return {"p_loss": probe_loss, "p_acc": self.probe_accuracy.result()}


class MomentumContrastiveModel(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
        momentum_coeff,
    ):
        super().__init__(
            contrastive_augmenter,
            classification_augmenter,
            encoder,
            projection_head,
            linear_probe,
        )
        self.momentum_coeff = momentum_coeff

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = tf.keras.models.clone_model(self.encoder)
        self.m_projection_head = tf.keras.models.clone_model(self.projection_head)

    @abstractmethod
    def contrastive_loss(
        self,
        projections_1,
        projections_2,
        m_projections_1,
        m_projections_2,
    ):
        pass

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            projections_1 = self.projection_head(self.encoder(augmented_images_1))
            projections_2 = self.projection_head(self.encoder(augmented_images_2))
            m_projections_1 = self.m_projection_head(self.m_encoder(augmented_images_1))
            m_projections_2 = self.m_projection_head(self.m_encoder(augmented_images_2))
            contrastive_loss = self.contrastive_loss(
                projections_1, projections_2, m_projections_1, m_projections_2
            )
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(projections_1, projections_2)

        preprocessed_images = self.classification_augmenter(labeled_images)
        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_accuracy.update_state(labels, class_logits)

        # the momentum networks are updated by exponential moving average
        for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
            m_weight.assign(
                self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight
            )
        for weight, m_weight in zip(
            self.projection_head.weights, self.m_projection_head.weights
        ):
            m_weight.assign(
                self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight
            )

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result(),
        }


class BarlowTwins(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
        redundancy_reduction_weight,
    ):
        super().__init__(
            contrastive_augmenter,
            classification_augmenter,
            encoder,
            projection_head,
            linear_probe,
        )
        # weighting coefficient between the two loss components
        self.redundancy_reduction_weight = redundancy_reduction_weight
        # its value differs from the paper, because the loss implementation has been
        # changed to be invariant to the encoder output dimensions (feature dim)

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = (
            projections_1 - tf.reduce_mean(projections_1, axis=0)
        ) / tf.math.reduce_std(projections_1, axis=0)
        projections_2 = (
            projections_2 - tf.reduce_mean(projections_2, axis=0)
        ) / tf.math.reduce_std(projections_2, axis=0)

        # the cross correlation of image representations should be the identity matrix
        batch_size = tf.shape(projections_1, out_type=tf.float32)[0]
        feature_dim = tf.shape(projections_1, out_type=tf.float32)[1]
        cross_correlation = (
            tf.matmul(projections_1, projections_2, transpose_a=True) / batch_size
        )
        squared_errors = (tf.eye(feature_dim) - cross_correlation) ** 2

        # invariance loss = average diagonal error
        # redundancy reduction loss = average off-diagonal error
        invariance_loss = (
            tf.reduce_sum(squared_errors * tf.eye(feature_dim)) / feature_dim
        )
        redundancy_reduction_loss = tf.reduce_sum(
            squared_errors * (1 - tf.eye(feature_dim))
        ) / (feature_dim * (feature_dim - 1))
        return (
            invariance_loss
            + self.redundancy_reduction_weight * redundancy_reduction_loss
        )


class SimCLR(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
        temperature,
    ):
        super().__init__(
            contrastive_augmenter,
            classification_augmenter,
            encoder,
            projection_head,
            linear_probe,
        )
        self.temperature = temperature

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # the temperature-scaled similarities are used as logits for cross-entropy
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
            from_logits=True,
        )
        return loss


class MoCo(MomentumContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
        momentum_coeff,
        temperature,
    ):
        super().__init__(
            contrastive_augmenter,
            classification_augmenter,
            encoder,
            projection_head,
            linear_probe,
            momentum_coeff,
        )
        self.temperature = temperature

        # no momentum-feature queue is implemented yet

    def contrastive_loss(
        self,
        projections_1,
        projections_2,
        m_projections_1,
        m_projections_2,
    ):
        # similar to the SimCLR loss, however it uses the momentum networks'
        # representations of the differently augmented views as targets
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        m_projections_1 = tf.math.l2_normalize(m_projections_1, axis=1)
        m_projections_2 = tf.math.l2_normalize(m_projections_2, axis=1)

        similarities_1 = (
            tf.matmul(projections_1, m_projections_2, transpose_b=True)
            / self.temperature
        )
        similarities_2 = (
            tf.matmul(projections_2, m_projections_1, transpose_b=True)
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities_1, similarities_2], axis=0),
            from_logits=True,
        )
        return loss
