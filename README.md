# Regularized Regression & Classification: OLS, Ridge, LASSO, Logistic

A from-scratch implementation and comparison of core supervised learning methods, applied to synthetic data and the MNIST handwritten digit dataset.

## Overview

This project implements and compares linear models for regression and classification, deriving the closed-form solutions analytically before validating them experimentally.

**Regression:**
- OLS (ordinary least squares), including polynomial feature expansion for non-linear data
- Ridge regression (L2 regularization)
- LASSO (L1 regularization)

**Classification:**
- Linear classifiers derived from OLS/Ridge risk minimization
- Logistic regression, tested under varying class balance and outlier proportions
- Multiclass logistic regression with L1 and L2 regularization, applied to MNIST digit classification

## Key Results

- **OLS vs. regularization**: OLS overfits in the presence of outliers and non-linear patterns; Ridge and LASSO both improve robustness by penalizing large coefficients, with LASSO additionally performing feature selection by zeroing out irrelevant weights.
- **Classification robustness**: OLS/Ridge-based classifiers are sensitive to class imbalance and outlier proportion, while logistic regression maintains stable decision boundaries even with skewed data.
- **MNIST digit classification**: Logistic regression with L2 regularization achieved strong diagonal accuracy in the confusion matrix, with residual confusion between visually similar digits (e.g. 4/9, 3/5). L1 regularization produced a sparser, more interpretable model with a modest accuracy trade-off.

## Tech Stack

Python, NumPy, scikit-learn, Matplotlib

## Team

Manâl Rhazza, Romane Pujol — ENSEIRB-MATMECA, PS204 (Statistics) course
