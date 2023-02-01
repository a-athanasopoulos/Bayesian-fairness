import numpy as np
import pandas as pd
import tensorflow as tf

from src.continuous.models.models import get_models_from_data
from src.continuous.tf.loss.fairness import get_fairness_loss
from src.continuous.tf.loss.utility import get_utility_loss, get_utility_metric
from src.continuous.tf.models.logistic_regression import LogisticRegressionTF


class BaseAlgorithm(object):

    def evaluate(self, **kwargs):
        raise NotImplemented("Implement evaluation functions")

    @staticmethod
    def update_policy(policy, dirichlet_belief, utility, l, lr, n_iter, **kwargs):
        raise NotImplemented("Implement evaluation functions")

    def fit(self, **kwargs):
        raise NotImplemented("Implement evaluation functions")


class ContinuousBaseAlgorithm(object):
    utility_tracker = tf.keras.metrics.Mean(name="utility")
    fairness_tracker = tf.keras.metrics.Mean(name="fairness")
    loss_tracker = tf.keras.metrics.Mean(name="loss")

    eval_utility_tracker = tf.keras.metrics.Mean(name="eval_utility")
    eval_fairness_tracker = tf.keras.metrics.Mean(name="eval_fairness")
    eval_loss_tracker = tf.keras.metrics.Mean(name="eval_loss")

    def __init__(self,
                 train_data,
                 test_data_and_models,
                 Z_atr,
                 X_atr,
                 Y_atr,
                 n_y,
                 n_z,
                 prior):
        self.train_data = train_data
        self.test_data_and_models = test_data_and_models
        self.Z_atr = Z_atr
        self.X_atr = X_atr
        self.Y_atr = Y_atr
        self.n_y = n_y
        self.n_z = n_z
        self.prior = prior

        self.initial_policy_weights = None
        self.final_policy_weights = None
        self.results = None
        self.run_parameters = {}
        self.policy_model = None
        self.optimizer = None

    def save_results(self, save_path):
        pd.DataFrame(self.initial_policy_weights[0]).to_csv(save_path + "/init_w.csv")
        pd.DataFrame([self.initial_policy_weights[1]]).to_csv(save_path + "/init_b.csv")
        pd.DataFrame(self.final_policy_weights[0]).to_csv(save_path + "/final_w.csv")
        pd.DataFrame([self.final_policy_weights[1]]).to_csv(save_path + "/final_b.csv")
        self.results.to_csv(save_path + "/results.csv")
        pd.DataFrame(self.run_parameters, index=["parameters"]).to_csv(save_path + "/run_parameters.csv")

    def convert_to_tensors(self, list_of_np):
        """
        converts a list of numpy to a list of tensors
        """
        list_of_tensors = [tf.convert_to_tensor(numpy_array) for numpy_array in list_of_np]
        return list_of_tensors

    @staticmethod
    def reset_trackers(trackers):
        for tracker in trackers:
            tracker.reset_states()

    @tf.function
    def eval_step(self, policy, data, lamba_parameter):
        x, y, Py, Pz_y, Py_x, Pz_yx = data

        y_pred = policy(x, training=False)
        utility = get_utility_metric(y_true=y, y_pred=y_pred)
        Pa_x = tf.stack([1 - y_pred, y_pred])
        Pa_x = tf.squeeze(Pa_x, axis=-1, name=None)
        fairness = get_fairness_loss(Pa_x, Py, Pz_y, Py_x, Pz_yx)
        loss = utility - lamba_parameter * fairness

        # update metrics
        self.eval_loss_tracker.update_state(loss)
        self.eval_utility_tracker.update_state(utility)
        self.eval_fairness_tracker.update_state(fairness)

        metrics = dict()
        metrics["eval_fairness_loss"] = self.eval_fairness_tracker.result()
        metrics["eval_utility"] = self.eval_utility_tracker.result()
        metrics["eval_loss"] = self.eval_loss_tracker.result()
        return metrics

    @tf.function
    def train_step(self, policy, data, lambda_parameter, optimizer):
        x, y, Py, Pz_y, Py_x, Pz_yx = data
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = policy(x, training=True)
            utility = get_utility_loss(y_true=y, y_pred=y_pred)
            Pa_x = tf.stack([1 - y_pred, y_pred])
            Pa_x = tf.squeeze(Pa_x, axis=-1, name=None)
            fairness = get_fairness_loss(Pa_x, Py, Pz_y, Py_x, Pz_yx)
            # loss = (1 - lambda_parameter) * utility + lambda_parameter * fairness
            loss = utility + lambda_parameter * fairness

            # Run backwards pass.
        optimizer.minimize(loss, policy.trainable_variables, tape=tape)
        # update metrics
        self.loss_tracker.update_state(loss)
        self.utility_tracker.update_state(get_utility_metric(y_true=y, y_pred=y_pred))
        self.fairness_tracker.update_state(fairness)

        metrics = {}
        metrics["fairness_loss"] = self.fairness_tracker.result()
        metrics["utility"] = self.utility_tracker.result()
        metrics["loss"] = self.loss_tracker.result()
        return metrics

    def compile(self, initial_policy_weights, learning_rate):
        self.initial_policy_weights = initial_policy_weights
        self.policy_model = LogisticRegressionTF(input_dim=len(self.X_atr))
        self.policy_model.set_weights(self.initial_policy_weights)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.run_parameters["lr"] = learning_rate

    def plot_history(self, save_path, show=0):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))
        # plot total
        ax1.plot(self.results["utility"], label="utility")
        ax1.plot(self.results["eval_utility"], label="eval utility")
        ax1.set_title("utility")
        ax1.legend()

        ax2.plot(self.results["fairness_loss"], label="fairness_loss")
        ax2.plot(self.results["eval_fairness_loss"], label="eval_fairness_loss")
        ax2.set_title("fairness")
        ax2.legend()

        if save_path:
            fig.savefig(save_path + f"/history.png")
        if show:
            fig.show()

        plt.close()


class ContinuousMarginalFairOptimization(ContinuousBaseAlgorithm):
    def fit(self, n_iter, lambda_parameter, **kwargs):
        self.run_parameters["n_iter"] = n_iter
        self.run_parameters["lambda_parameter"] = lambda_parameter
        # get train data
        x = tf.convert_to_tensor(self.train_data[self.X_atr].values)
        y = tf.reshape(tf.convert_to_tensor(self.train_data[self.Y_atr].values), (-1, 1))
        train_model = get_models_from_data(data=self.train_data,
                                           X_atr=self.X_atr, Y_atr=self.Y_atr, Z_atr=self.Z_atr,
                                           n_y=self.n_y, n_z=self.n_z)

        # test data
        test_data_list = (tf.convert_to_tensor(p, dtype="float32") for p in
                          self.test_data_and_models)

        # loop
        history = []
        eval_history = []
        for iter in range(n_iter):
            tf_Py, tf_Pz_y, tf_Py_x, tf_Pz_yx = (tf.convert_to_tensor(p, dtype="float32") for p in train_model)

            # training step
            step_results = self.train_step(policy=self.policy_model,
                                           data=(x, y, tf_Py, tf_Pz_y, tf_Py_x, tf_Pz_yx),
                                           lambda_parameter=lambda_parameter,
                                           optimizer=self.optimizer)
            step_results = {key: step_results[key].numpy() for key in step_results.keys()}
            history += [step_results]

            # evaluation step

            eval_results = self.eval_step(policy=self.policy_model,
                                          data=test_data_list,
                                          lamba_parameter=lambda_parameter)
            eval_results = {key: eval_results[key].numpy() for key in eval_results.keys()}
            eval_history += [eval_results]

            self.reset_trackers([self.loss_tracker, self.utility_tracker, self.fairness_tracker,
                                 self.eval_loss_tracker, self.eval_utility_tracker, self.eval_fairness_tracker])
            # print(f"--- Step : {iter + 1} \n  ------- {step_results}")

        pd_history = pd.DataFrame(history)
        pd_eval_history = pd.DataFrame(eval_history)
        self.results = pd.concat([pd_history, pd_eval_history], axis=1)
        self.final_policy_weights = self.policy_model.get_weights()


class ContinuousBootstrapFairOptimization(ContinuousBaseAlgorithm):
    def fit(self, n_iter, lambda_parameter, bootstrap_models, **kwargs):
        self.run_parameters["n_iter"] = n_iter
        self.run_parameters["lambda_parameter"] = lambda_parameter
        self.run_parameters["bootstrap_models"] = bootstrap_models
        # get train data
        train_models = []
        bootstrap_datasets = []
        for n in range(bootstrap_models):
            bootstrap_dataset = self.train_data.sample(frac=1.0, replace=True).reset_index(drop=True)
            bootstrap_datasets += [bootstrap_dataset]
            train_models += [get_models_from_data(data=bootstrap_dataset,
                                                  X_atr=self.X_atr, Y_atr=self.Y_atr, Z_atr=self.Z_atr,
                                                  n_y=self.n_y, n_z=self.n_z)]

        # test data
        test_data_list = (tf.convert_to_tensor(p, dtype="float32") for p in
                          self.test_data_and_models)

        # loop
        history = []
        eval_history = []
        for iter in range(n_iter):
            # model_sample = iter % bootstrap_models
            model_sample = np.random.choice(range(bootstrap_models))
            x = tf.convert_to_tensor(bootstrap_datasets[model_sample][self.X_atr].values)
            y = tf.reshape(tf.convert_to_tensor(bootstrap_datasets[model_sample][self.Y_atr].values), (-1, 1))
            tf_Py, tf_Pz_y, tf_Py_x, tf_Pz_yx = (tf.convert_to_tensor(p, dtype="float32") for p in
                                                 train_models[model_sample])

            # training step
            step_results = self.train_step(policy=self.policy_model,
                                           data=(x, y, tf_Py, tf_Pz_y, tf_Py_x, tf_Pz_yx),
                                           lambda_parameter=lambda_parameter,
                                           optimizer=self.optimizer)
            step_results = {key: step_results[key].numpy() for key in step_results.keys()}
            history += [step_results]

            # evaluation step

            eval_results = self.eval_step(policy=self.policy_model,
                                          data=test_data_list,
                                          lamba_parameter=lambda_parameter)
            eval_results = {key: eval_results[key].numpy() for key in eval_results.keys()}
            eval_history += [eval_results]

            self.reset_trackers([self.loss_tracker, self.utility_tracker, self.fairness_tracker,
                                 self.eval_loss_tracker, self.eval_utility_tracker, self.eval_fairness_tracker])
            # print(f"--- Step : {iter + 1} \n  ------- {step_results}")

        pd_history = pd.DataFrame(history)
        pd_eval_history = pd.DataFrame(eval_history)
        self.results = pd.concat([pd_history, pd_eval_history], axis=1)
        self.final_policy_weights = self.policy_model.get_weights()
