import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter


class LogisticRegressionTF(tf.keras.Model):
    """
    logistic regresion model , tensorflow
    """

    def __init__(self, input_dim, Py, Pz_y, Py_x, Pz_yx):
        super(LogisticRegressionTF, self).__init__()
        w_init = tf.random_normal_initializer()
        self.Py = Py
        self.Pz_y = Pz_y
        self.Py_x = Py_x
        self.Pz_yx = Pz_yx
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

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
            fairness_loss =
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)
