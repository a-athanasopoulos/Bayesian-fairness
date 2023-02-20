# Approximate Inference for the Bayesian Fairness Framework

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

the code of the project is on src containing the following modules:
1. continuous: the base code for the continuous experiment
2. discrete: the base code for the Discrete experiment
3. experiments: the experiments of the paper
4. notebooks: jupyter notebook to visualise results for a particular experiment

In the following steps we describe how one can reproduce the results of the paper.

#### Requirements
The instruction are for linux based system with python 3.9

#### Setting Environment

To setup the python virtual environment run, navigate to the project directory and run the following command from a terminal

> cd path_to_repo/Bayesian-fairness  
> python3 -m venv bayesian-fairness_env  
> source bayesian-fairness_env/bin/activate  
> pip3 install -r requirements.txt  

#### Run Experiments
To reproduce the result run the following code
> cd path_to_repo/Bayesian-fairness  
> source bayesian-fairness_env/bin/activate  
> python3.9 -m src.experiments.reproduce  

