# Bayesian-fairness

The current repository contains the code to implement the "Approximate Inference for the Bayesian Fairness
Framework" paper in python.

### Abstract

As the impact of Artificial Intelligence systems and applications on everyday life increases, algorithmic
fairness undoubtedly constitutes one of the major problems in our modern society. In the current paper,
we extend the work of Dimitrakakis et al. on Bayesian fairness that incorporates models uncertainty
to achieve fairness, proposing a practical algorithm with the aim to scale the framework for more general
applications. We begin by applying the bootstrap technique as a scalable alternative to approximate
the parameter posterior distribution of the fully Bayesian viewpoint. To make the Bayesian fairness
framework applicable to more general data settings, we define an empirical formulation suitable for
the continuous case. We empirically demonstrate the potential of the framework from an extensive
evaluation study on a real dataset and different decision settings

### Overview
The repository contains the data and the code for reproducing the results of the  "Approximate Inference for the Bayesian Fairness
Framework".

The project in organized in the following main directories:
1. data: directory contains the data
2. results: directory to save results
3. src: directory with the code

In the following steps we describe how one can reproduce the results of the papper.

#### Requirements
The instruction are for linux based system with python 3.9

#### Setting Environment

To setup the python virtual environment run, navigate to the project directory and run the following command from a terminal

> python3 -m venv bayesian-fairness_env
> source bayesian-fairness_env/bin/activate
> pip3 install -r requirements.txt

#### Run Experiments

The experiment 

#### 1. implementation of bayesian fairness
* paper: https://ojs.aaai.org/index.php/AAAI/article/view/3824  
* code_path :  src > experiments > bayesian_fairness  
* results_path: results > bayesian_fairness
* notebooks: src> notebooks > bayesian_fairness