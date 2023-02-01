import tensorflow as tf


def get_fairness_loss_paper(Pa_x, Py, Pz_y, Py_x, Pz_yx):
    # compute delta
    term_1 = (Pz_yx / tf.expand_dims(Pz_y, axis=-1)) - 1
    term_2 = Py_x / tf.expand_dims(Py, axis=-1)
    delta = term_1 * tf.expand_dims(term_2, axis=0)

    # compute c
    exp_Pa_x = tf.expand_dims(tf.expand_dims(Pa_x, axis=0), axis=0)
    exp_delta = tf.expand_dims(delta, axis=-1)
    c = tf.linalg.matmul(exp_Pa_x,
                         exp_delta)

    # compute fairness
    fairness = tf.norm(c, ord=1) / Py_x.shape[1]
    return fairness


def get_fairness_loss(Pa_x, Py, Pz_y, Py_x, Pz_yx):
    # compute delta
    term_1 = Pz_yx - tf.expand_dims(Pz_y, axis=-1)
    term_2 = Py_x / tf.expand_dims(Py, axis=-1)
    delta = term_1 * tf.expand_dims(term_2, axis=0)

    # compute c
    exp_Pa_x = tf.expand_dims(tf.expand_dims(Pa_x, axis=0), axis=0)
    exp_delta = tf.expand_dims(delta, axis=-1)
    c = tf.linalg.matmul(exp_Pa_x,
                         exp_delta)

    # compute fairness
    fairness = tf.norm(c, ord=1) / Py_x.shape[1]
    return fairness
