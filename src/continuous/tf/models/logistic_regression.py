import tensorflow as tf


class LogisticRegressionTF(tf.keras.Model):
    """
    logistic regression model , tensorflow
    """

    def __init__(self, input_dim):
        super(LogisticRegressionTF, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, 1), dtype="float32"),
            trainable=True,
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(), dtype="float32"),
            trainable=True
        )

    def call(self, inputs, **kwargs):
        return tf.nn.sigmoid(tf.matmul(inputs, self.w) + self.b)
