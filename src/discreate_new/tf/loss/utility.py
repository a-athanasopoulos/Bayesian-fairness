import tensorflow as tf


def get_utility_loss(utility_matrix, policy, Pxy):
    m = tf.matmul(policy, Pxy)
    utility = tf.math.reduce_sum(tf.math.multiply(utility_matrix, m))
    return - utility
