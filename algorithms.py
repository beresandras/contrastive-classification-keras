import tensorflow as tf

from tensorflow import keras
from models import ContrastiveModel, MomentumContrastiveModel


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


class NNCLR(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
        temperature,
        queue_size,
    ):
        super().__init__(
            contrastive_augmenter,
            classification_augmenter,
            encoder,
            projection_head,
            linear_probe,
        )
        self.temperature = temperature

        feature_dimensions = encoder.output_shape[1]
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

    def nearest_neighbour(self, projections):
        # highest cosine similarity == lowest L2 distance, for L2 normalized features
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )

        # hard nearest-neighbours
        nn_projections = tf.gather(
            self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
        )

        # straight-through gradient estimation
        # paper used stop gradient, however it helps performance at this scale
        return projections + tf.stop_gradient(nn_projections - projections)

    def contrastive_loss(self, projections_1, projections_2):
        # similar to the SimCLR loss, however we take the nearest neighbours of a set
        # of projections from a feature queue
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities_1_2 = (
            tf.matmul(
                self.nearest_neighbour(projections_1), projections_2, transpose_b=True
            )
            / self.temperature
        )
        similarities_2_1 = (
            tf.matmul(
                self.nearest_neighbour(projections_2), projections_1, transpose_b=True
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities_1_2, similarities_2_1], axis=0),
            from_logits=True,
        )

        # feature queue update
        self.feature_queue.assign(
            tf.concat([projections_1, self.feature_queue[:-batch_size]], axis=0)
        )
        return loss


class DCCLR(ContrastiveModel):
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
        # a modified InfoNCE loss, which should provide better performance at
        # lower batch sizes

        # cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # the similarities of the positives (the main diagonal) are masked and
        # are not included in the softmax normalization
        batch_size = tf.shape(projections_1)[0]
        decoupling_mask = 1.0 - tf.eye(batch_size)
        decoupled_similarities = decoupling_mask * tf.exp(similarities)

        loss = tf.reduce_mean(
            -tf.linalg.diag_part(similarities)
            + tf.math.log(
                tf.reduce_sum(decoupled_similarities, axis=0)
                + tf.reduce_sum(decoupled_similarities, axis=1)
            )
        )
        # the sum along the two axes should be put in separate log-sum-exp
        # expressions according to the paper, this however achieves slightly
        # higher performance at this scale

        return loss


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
        target_cross_correlation = tf.eye(feature_dim)
        squared_errors = (target_cross_correlation - cross_correlation) ** 2

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


class HSICTwins(ContrastiveModel):
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
        # a modified BarlowTwins loss, derived from Hilbert-Schmidt Independence
        # Criterion maximization, the only difference is the target cross correlation

        projections_1 = (
            projections_1 - tf.reduce_mean(projections_1, axis=0)
        ) / tf.math.reduce_std(projections_1, axis=0)
        projections_2 = (
            projections_2 - tf.reduce_mean(projections_2, axis=0)
        ) / tf.math.reduce_std(projections_2, axis=0)

        # the cross correlation of image representations should be 1 along the diagonal
        # and -1 everywhere else
        batch_size = tf.shape(projections_1, out_type=tf.float32)[0]
        feature_dim = tf.shape(projections_1, out_type=tf.float32)[1]
        cross_correlation = (
            tf.matmul(projections_1, projections_2, transpose_a=True) / batch_size
        )
        target_cross_correlation = 2.0 * tf.eye(feature_dim) - 1.0
        squared_errors = (target_cross_correlation - cross_correlation) ** 2

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


class TWIST(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
    ):
        super().__init__(
            contrastive_augmenter,
            classification_augmenter,
            encoder,
            projection_head,
            linear_probe,
        )

    def contrastive_loss(self, projections_1, projections_2):
        # a probabilistic, hyperparameter- and negative-free loss

        # batch normalization before softmax operation
        projections_1 = (
            projections_1 - tf.reduce_mean(projections_1, axis=0)
        ) / tf.math.reduce_std(projections_1, axis=0)
        projections_2 = (
            projections_2 - tf.reduce_mean(projections_2, axis=0)
        ) / tf.math.reduce_std(projections_2, axis=0)

        probabilities_1 = keras.activations.softmax(projections_1)
        probabilities_2 = keras.activations.softmax(projections_2)

        mean_probabilities_1 = tf.reduce_mean(probabilities_1, axis=0)
        mean_probabilities_2 = tf.reduce_mean(probabilities_2, axis=0)

        # cross-entropy(1,2): KL-div(1,2) (consistency) + entropy(1) (sharpness)
        # -cross-entropy(mean1,mean1): -entropy(mean1) (diversity)
        loss = keras.losses.categorical_crossentropy(
            tf.concat([probabilities_1, probabilities_2], axis=0),
            tf.concat([probabilities_2, probabilities_1], axis=0),
        ) - keras.losses.categorical_crossentropy(
            tf.concat([mean_probabilities_1, mean_probabilities_2], axis=0),
            tf.concat([mean_probabilities_1, mean_probabilities_2], axis=0),
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
        queue_size,
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

        feature_dimensions = encoder.output_shape[1]
        self.feature_queue = tf.Variable(
            tf.math.l2_normalize(
                tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
            ),
            trainable=False,
        )

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

        similarities_1_2 = (
            tf.matmul(
                projections_1,
                tf.concat((m_projections_2, self.feature_queue), axis=0),
                transpose_b=True,
            )
            / self.temperature
        )
        similarities_2_1 = (
            tf.matmul(
                projections_2,
                tf.concat((m_projections_1, self.feature_queue), axis=0),
                transpose_b=True,
            )
            / self.temperature
        )

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        loss = keras.losses.sparse_categorical_crossentropy(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities_1_2, similarities_2_1], axis=0),
            from_logits=True,
        )

        # feature queue update
        self.feature_queue.assign(
            tf.concat(
                [
                    m_projections_1,
                    m_projections_2,
                    self.feature_queue[: -(2 * batch_size)],
                ],
                axis=0,
            )
        )
        return loss


class DINO(MomentumContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
        momentum_coeff,
        temperature,
        sharpening,
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
        self.sharpening = sharpening

    def contrastive_loss(
        self,
        projections_1,
        projections_2,
        m_projections_1,
        m_projections_2,
    ):
        # this loss does not use any negatives, needs centering + sharpening + momentum
        # to avoid collapse

        # l2-normalization is part of the projection head in the original implementation
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        m_projections_1 = tf.math.l2_normalize(m_projections_1, axis=1)
        m_projections_2 = tf.math.l2_normalize(m_projections_2, axis=1)

        center = tf.reduce_mean(
            tf.concat([m_projections_1, m_projections_2], axis=0), axis=0, keepdims=True
        )
        target_probabilities_1 = keras.activations.softmax(
            (m_projections_1 - center) / (self.sharpening * self.temperature)
        )
        target_probabilities_2 = keras.activations.softmax(
            (m_projections_2 - center) / (self.sharpening * self.temperature)
        )

        pred_probabilities_1 = keras.activations.softmax(
            projections_1 / self.temperature
        )
        pred_probabilities_2 = keras.activations.softmax(
            projections_2 / self.temperature
        )

        loss = keras.losses.categorical_crossentropy(
            tf.concat([target_probabilities_1, target_probabilities_2], axis=0),
            tf.concat([pred_probabilities_2, pred_probabilities_1], axis=0),
        )
        return loss