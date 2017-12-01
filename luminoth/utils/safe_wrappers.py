import tensorflow as tf


def safe_softmax(inputs, name='safe_softmax'):
    """Compute softmax if there are elements. Return an empty Tensor if not.

    Currently (last tested TF 1.4) tf.nn.softmax breaks if the Tensor has None
    shape in the first dimension and is empty.
    """
    with tf.name_scope(name):
        input_length = tf.shape(inputs, name='input_shape')[0]
        safety_condition = tf.greater(input_length, 0, name='condition')
        return tf.cond(
            safety_condition,
            true_fn=lambda: tf.nn.softmax(inputs),
            false_fn=lambda: tf.constant([], dtype=inputs.dtype),
        )
