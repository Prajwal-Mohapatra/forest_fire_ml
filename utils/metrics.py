import tensorflow as tf
import keras
import keras.backend as K

@keras.saving.register_keras_serializable()
def iou_score(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Intersection over Union metric for binary segmentation with configurable threshold"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Dice coefficient for binary segmentation with configurable threshold"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@keras.saving.register_keras_serializable()
def fire_recall(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Fire-specific recall metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    possible_positives = tf.reduce_sum(y_true)
    return (true_positives + smooth) / (possible_positives + smooth)

@keras.saving.register_keras_serializable()
def fire_precision(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Fire-specific precision metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    return (true_positives + smooth) / (predicted_positives + smooth)

@keras.saving.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.6):
    """Focal loss for handling class imbalance - Updated alpha from 0.25 to 0.6 for better balance"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss = -focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed
