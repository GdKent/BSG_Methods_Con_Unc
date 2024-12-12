# Inexact Bilevel Stochastic Gradient Methods for Constrained and Unconstrained Lower-Level Problems (BSG-N-FD & BSG-1)

This is the numerical implementation accompanying the paper "Inexact Bilevel Stochastic Gradient Methods for Constrained and Unconstrained Lower-Level Problems". Please refer to the [paper](https://arxiv.org/abs/2110.00604) for more details.

## 1. Software Requirements

The code was implemented in Python 3.7.12 and requires the following libraries:

+ cudatoolkit 11.3.1
+ numpy 1.21.5
+ pandas 1.3.5
+ scikit-learn 1.0.2
+ torch 1.10.0
+ torchvision 0.11.0
+ torchauio 0.10.0
+ scipy 1.7.3
+ seaborn 0.12.2
+ matplotlib 3.5.3


## 2. Python Scripts

The .py files can be broken up into three categories: a "function" file, a "bilevel_solver" file, and "driver" files.

+ __"bilevel_solver"__ file: This file contains all of the code for the implementations of the bilevel algorithms BSG-N-FD, BSG-H, BSG-1, StocBiO, DARTS, and SIGD.
+ __"functions"__ file: This file contains all the the code that defines the different types of problems that were tested, i.e., both deterministic and stochastic versions of a general synthetic problem (either with no constraints, linear constraints in y, or quadratic constraints in x and y) or a continual learning problem defined using the CIFAR-10 dataset (either with or without constraints).
+ __"Driver"__ files: These files contain all of the code for running the specific experiments to generate all of the plots for the figures that are displayed in the paper.

## In case you cite our work, please refer to the paper:

T. Giovannelli, G. Kent, and L. N. Vicente. Inexact Bilevel Stochastic Gradient Methods for Constrained and Unconstrained Lower-Level Problems. ISE Technical Report 21T-025, Lehigh University, October 2021.



