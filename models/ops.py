import tensorflow as tf


def batch_norm(input,
               is_training,
               momentum=0.9,
               epsilon=1e-5,
               in_place_update=True,
               scope=None):
    if in_place_update:
        return tf.contrib.layers.batch_norm(
            input,
            decay=momentum,
            center=True,
            scale=True,
            epsilon=epsilon,
            updates_collections=None,
            is_training=is_training,
            scope=scope)
    else:
        return tf.contrib.layers.batch_norm(
            input,
            decay=momentum,
            center=True,
            scale=True,
            epsilon=epsilon,
            is_training=is_training,
            scope=scope)
