# Bank-Marketing-Campaign-Optimization
A machine learning solution for optimizing bank marketing campaigns by predicting successful customer contacts using Random Forest classification.

# Project Overview
This project develops a predictive model to identify which bank customers are most likely to respond positively to term deposit offers, helping maximize marketing efficiency and profitability.

# Business Problem
Banks conduct marketing campaigns to promote term deposits, but contacting all customers is costly and inefficient. This solution predicts successful contacts to:

* Reduce marketing costs
* Increase conversion rates
* Maximize return on investment

# Dataset
* File: `3625_assign2_data_train.csv`
* Target Variable: `success` (whether customer subscribed to term deposit)
* Features: Various customer attributes and campaign data

# Technical Implementation
## Core Features
* Data Preprocessing: Handling class imbalance using SMOTE and RandomUnderSampler
* Model Training: Random Forest Classifier with optimized hyperparameters
* Performance Evaluation: Comprehensive metrics including confusion matrix and classification reports
* Profit Analysis: Cost-benefit analysis of marketing campaigns

# Key Functions
`identify_customers()`
* Loads pre-trained model and makes predictions
* Calculates revenue, costs, and profit
* Outputs performance metrics and business insights\

`random_forest()`
* Trains Random Forest model with SMOTE oversampling
* Saves trained model using pickle
* Evaluates model performance on test data

`under_over_sample()`
* Handles class imbalance using either:
** SMOTE (Synthetic Minority Over-sampling Technique)
** RandomUnderSample

# Business Logic
* Contact Cost: $10 per customer
* Revenue: 4% of customer balance for successful conversions
* Profit Calculation: `Total Revenue - (Contacts × $10)`

# Key Insights
* Feature Importance: Previous outcomes, housing status, and contact method significantly impact success
* Class Imbalance: Addressed through strategic sampling techniques
* Profit Optimization: Targeted approach significantly outperforms blanket marketing

This solution demonstrates practical application of machine learning in financial marketing, balancing technical accuracy with business profitability.
