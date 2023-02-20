import numpy as np
import pandas as pd
import tensorflow as tf

from src.discreate.tf.models.logistic_regression import LogisticRegressionTF
from src.discreate.models.boostrap_model import BoostrapModel
from src.discreate.models.dirichlet_model import DirichletModel
from src.discreate.tf.loss.fairness import get_fairness
from src.discreate.tf.loss.utility import get_utility_loss
from src.discreate.utils.model import get_delta


class BaseAlgorithm(object):

    def evaluate(self, **kwargs):
        raise NotImplemented("Implement evaluation functions")

    @staticmethod
    def update_policy(policy, dirichlet_belief, utility, l, lr, n_iter, **kwargs):
        raise NotImplemented("Implement evaluation functions")

    def fit(self, **kwargs):
        raise NotImplemented("Implement evaluation functions")


class DiscreteBaseAlgorithm(object):
    utility_tracker = tf.keras.metrics.Mean(name="utility")
    fairness_tracker = tf.keras.metrics.Mean(name="fairness")
    loss_tracker = tf.keras.metrics.Mean(name="loss")

    eval_utility_tracker = tf.keras.metrics.Mean(name="eval_utility")
    eval_fairness_tracker = tf.keras.metrics.Mean(name="eval_fairness")
    eval_loss_tracker = tf.keras.metrics.Mean(name="eval_loss")

    def __init__(self,
                 train_data,
                 test_models,
                 Z_atr,
                 X_atr,
                 Y_atr,
                 n_x,
                 n_y,
                 n_z,
                 prior):
        self.train_data = train_data
        self.test_models = test_models
        self.test_delta = get_delta(test_models.Px_y, test_models.Px_yz, test_models.Pz_y)
        self.Z_atr = Z_atr
        self.X_atr = X_atr
        self.Y_atr = Y_atr
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.prior = prior

        self.initial_policy_weights = None
        self.final_policy_weights = None
        self.results = None
        self.run_parameters = {}
        self.policy_model = None
        self.optimizer = None
        self.U = tf.eye(self.n_y)

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
        x, model_delta, p_xy = data

        y_pred = policy(x, training=False)
        Pa_x = tf.stack([1 - y_pred, y_pred])
        Pa_x = tf.squeeze(Pa_x, axis=-1, name=None)
        utility = get_utility_loss(utility_matrix=self.U, policy=Pa_x, Pxy=p_xy)
        fairness = get_fairness(policy=Pa_x, model_delta=model_delta)
        loss = utility + lamba_parameter * fairness
        loss = - loss
        # update metrics
        self.eval_loss_tracker.update_state(loss)
        self.eval_utility_tracker.update_state(- utility)
        self.eval_fairness_tracker.update_state(fairness)

        metrics = dict()
        metrics["eval_fairness_loss"] = self.eval_fairness_tracker.result()
        metrics["eval_utility"] = self.eval_utility_tracker.result()
        metrics["eval_loss"] = self.eval_loss_tracker.result()
        return metrics

    @tf.function
    def train_step(self, policy, data, lambda_parameter, optimizer):
        x, model_delta, p_xy = data
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = policy(x, training=True)
            Pa_x = tf.stack([1 - y_pred, y_pred])
            Pa_x = tf.squeeze(Pa_x, axis=-1, name=None)
            utility = get_utility_loss(utility_matrix=self.U, policy=Pa_x, Pxy=p_xy)
            fairness = get_fairness(policy=Pa_x, model_delta=model_delta)
            # loss = (1 - lambda_parameter) * utility + lambda_parameter * fairness
            loss = utility + lambda_parameter * fairness

        # Run backwards pass.
        optimizer.minimize(loss, policy.trainable_variables, tape=tape)

        # update metrics
        self.loss_tracker.update_state(loss)
        self.utility_tracker.update_state(-utility)
        self.fairness_tracker.update_state(fairness)

        metrics = {}
        metrics["fairness_loss"] = self.fairness_tracker.result()
        metrics["utility"] = self.utility_tracker.result()
        metrics["loss"] = self.loss_tracker.result()
        return metrics

    def compile(self, initial_policy_weights, learning_rate):
        self.initial_policy_weights = initial_policy_weights
        self.policy_model = LogisticRegressionTF(input_dim=self.n_x)
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


class DiscreteMarginalFairOptimization(DiscreteBaseAlgorithm):
    def fit(self, n_iter, lambda_parameter, **kwargs):
        self.run_parameters["n_iter"] = n_iter
        self.run_parameters["lambda_parameter"] = lambda_parameter

        # get marginal model
        belief = DirichletModel(n_x=self.n_x, n_y=self.n_y, n_z=self.n_z, prior=self.prior)
        belief.update_posterior_belief(self.train_data)
        model = belief.get_marginal_model()
        model_delta = get_delta(model.Px_y, model.Px_yz, model.Pz_y)

        x = tf.one_hot(range(self.n_x), self.n_x)
        # tf.reshape(tf.convert_to_tensor(range(self.n_x)), (-1, 1))

        # test data
        tf_test_delta = tf.convert_to_tensor(self.test_delta, dtype="float32")
        tf_test_p_xy = tf.convert_to_tensor(self.test_models.Pxy, dtype="float32")

        # loop
        history = []
        eval_history = []
        for iter in range(n_iter):
            tf_model_delta = tf.convert_to_tensor(model_delta, dtype="float32")
            tf_p_xy = tf.convert_to_tensor(model.Pxy, dtype="float32")
            # training step
            step_results = self.train_step(policy=self.policy_model,
                                           data=(x, tf_model_delta, tf_p_xy),
                                           lambda_parameter=lambda_parameter,
                                           optimizer=self.optimizer)
            step_results = {key: step_results[key].numpy() for key in step_results.keys()}
            history += [step_results]

            # evaluation step

            eval_results = self.eval_step(policy=self.policy_model,
                                          data=(x, tf_test_delta, tf_test_p_xy),
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


class DiscreteBootstrapFairOptimization(DiscreteBaseAlgorithm):
    def fit(self, n_iter, lambda_parameter, bootstrap_models, **kwargs):
        self.run_parameters["n_iter"] = n_iter
        self.run_parameters["lambda_parameter"] = lambda_parameter
        self.run_parameters["bootstrap_models"] = bootstrap_models
        # test data
        tf_test_delta = tf.convert_to_tensor(self.test_delta, dtype="float32")
        tf_test_p_xy = tf.convert_to_tensor(self.test_models.Pxy, dtype="float32")

        # get train data
        x = tf.one_hot(range(self.n_x), self.n_x)
        # tf.reshape(tf.convert_to_tensor(range(self.n_x)), (-1, 1))

        train_models = []
        model_deltas = []
        for m in range(bootstrap_models):
            model = BoostrapModel(n_x=self.n_x, n_y=self.n_y, n_z=self.n_z).sample_model(self.train_data)
            train_models += [model]
            model_deltas += [get_delta(model.Px_y, model.Px_yz, model.Pz_y)]

        # loop
        history = []
        eval_history = []
        for iter in range(n_iter):
            # model_sample = iter % bootstrap_models
            model_sample = np.random.choice(range(bootstrap_models))
            tf_model_delta = tf.convert_to_tensor(model_deltas[model_sample], dtype="float32")
            tf_p_xy = tf.convert_to_tensor(train_models[model_sample].Pxy, dtype="float32")

            # training step
            step_results = self.train_step(policy=self.policy_model,
                                           data=(x, tf_model_delta, tf_p_xy),
                                           lambda_parameter=lambda_parameter,
                                           optimizer=self.optimizer)
            step_results = {key: step_results[key].numpy() for key in step_results.keys()}
            history += [step_results]

            # evaluation step

            eval_results = self.eval_step(policy=self.policy_model,
                                          data=(x, tf_test_delta, tf_test_p_xy),
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


class DiscreteBayesianFairOptimization(DiscreteBaseAlgorithm):
    def fit(self, n_iter, lambda_parameter, n_model, **kwargs):
        self.run_parameters["n_iter"] = n_iter
        self.run_parameters["lambda_parameter"] = lambda_parameter
        self.run_parameters["n_model"] = n_model
        # test data
        tf_test_delta = tf.convert_to_tensor(self.test_delta, dtype="float32")
        tf_test_p_xy = tf.convert_to_tensor(self.test_models.Pxy, dtype="float32")

        # get train data
        x = tf.one_hot(range(self.n_x), self.n_x)
        # tf.reshape(tf.convert_to_tensor(range(self.n_x)), (-1, 1))

        belief = DirichletModel(n_x=self.n_x, n_y=self.n_y, n_z=self.n_z, prior=self.prior)
        belief.update_posterior_belief(self.train_data)
        train_models = []
        model_deltas = []
        for m in range(n_model):
            model = belief.sample_model()
            train_models += [model]
            model_deltas += [get_delta(model.Px_y, model.Px_yz,  model.Pz_y)]

        # loop
        history = []
        eval_history = []
        for iter in range(n_iter):
            # model_sample = iter % bootstrap_models
            model_sample = np.random.choice(range(n_model))
            tf_model_delta = tf.convert_to_tensor(model_deltas[model_sample], dtype="float32")
            tf_p_xy = tf.convert_to_tensor(train_models[model_sample].Pxy, dtype="float32")

            # training step
            step_results = self.train_step(policy=self.policy_model,
                                           data=(x, tf_model_delta, tf_p_xy),
                                           lambda_parameter=lambda_parameter,
                                           optimizer=self.optimizer)
            step_results = {key: step_results[key].numpy() for key in step_results.keys()}
            history += [step_results]

            # evaluation step

            eval_results = self.eval_step(policy=self.policy_model,
                                          data=(x, tf_test_delta, tf_test_p_xy),
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
