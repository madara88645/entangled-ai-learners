import tensorflow as tf
import math

def compute_entangled_loss(y_true, y_pred_self, y_pred_other, base_loss_fn=None, entangle_weight=0.01):
    if base_loss_fn is None:
        base_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    kl_div = tf.keras.losses.KLDivergence()
    base_loss = base_loss_fn(y_true, y_pred_self)
    entangled_part = kl_div(y_pred_self, y_pred_other)
    return base_loss + entangle_weight * entangled_part

def get_lambda(epoch, max_epochs=30, max_lambda=0.05):
    return (epoch / max_epochs) * max_lambda
