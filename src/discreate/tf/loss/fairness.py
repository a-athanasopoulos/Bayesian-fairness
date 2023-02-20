import tensorflow as tf


def get_fairness(policy, model_delta):
    expand_Pa_x = tf.expand_dims(tf.expand_dims(policy, axis=0), axis=0)
    expand_delta = tf.expand_dims(model_delta, axis=-1)
    c = tf.linalg.matmul(expand_Pa_x,
                         expand_delta)
    fairness = tf.norm(c, ord=1)
    return fairness
