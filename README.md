# deep-learning-challenge
Module 21

## Overview of the Analysis

In this analysis, the goal was to predict the success of charity applications using a deep learning model. Below is a detailed overview of the dataset, methodology, and results of the analysis.

## Dataset Overview

* Size: The dataset contains 34,299 entries.

* Key Columns:
  - Target Variable: IS_SUCCESSFUL (indicates whether an application was successful)
  - Feature Variables: Includes variables such as STATUS, ASK_AMT, various APPLICATION_TYPE_* columns, and others like ORGANIZATION_Trust and SPECIAL_CONSIDERATIONS_Y.

* Exclusions: The EIN and NAME columns were removed from the dataset during preprocessing as they were not useful for modeling.

## Methodology

* Machine Learning Model: A deep neural network was used to predict the success of the applications.

* Process:
  - Data Reading: The data was read from a charity_data.csv file into a Pandas DataFrame.
  - Data Preprocessing: Binned the CLASSIFICATION and APPLICATION_TYPE columns to group less frequent values.
  - Data Splitting: The data was split into training and testing sets using train_test_split.
  - Scaling: Features were scaled to ensure better model performance.
  - Model Compilation and Training:
        . Multiple attempts were made with different configurations of hidden layers, neurons, and epochs to optimize the model's performance.
        . Different activation functions and optimization methods were experimented with.
  - Model Evaluation: The model's performance was evaluated on the test set, and the results were saved and exported to an HDF5 file for later use.

## Results

* Data Preprocessing:
  - The target variable IS_SUCCESSFUL was identified, and irrelevant features like EIN and NAME were removed.
  - Binning was applied to the CLASSIFICATION and APPLICATION_TYPE columns to consolidate categories with low occurrences.

* Model Training and Evaluation:
 - A variety of attempts were made to improve the model's performance:
 - Dropped more columns, including EIN and NAME.
 - Adjusted the binning ranges for the CLASSIFICATION and APPLICATION_TYPE columns to group more values (e.g., from <1000 to <1500).
 - Increased the number of hidden layers from 2 to 4 layers, and modified the number of neurons in each layer (e.g., from 80, 30 to 100, 70, 30).
 - Experimented with increasing epochs from 100 to 150 to give the model more time to learn.

* Despite extensive experimentation, the model achieved an accuracy of approximately 72.5% with a loss of 0.56 on the test set.

## Summary of Findings

The deep learning model achieved 
 - in Starter_Code_1: an accuracy of 72.5% with a loss of 0.56 
 - in AlphabetSoupCharity_Optimization_1: an accuracy of 72.4% with a loss of 0.55 
 - in AlphabetSoupCharity_Optimization_2: an accuracy of 71.95% with a loss of 0.57 
 on the test data. While this indicates the model can make reasonable predictions, the results suggest that there is room for optimization or the use of a different model that may perform better for this task.

The deep learning approach, although valuable, may not always be the best for tabular data like this. Models such as Random Forest or Gradient Boosting Machines (GBM), which are better suited for structured data, could potentially achieve better results without the need for extensive tuning of hyperparameters. These models are often less prone to overfitting and handle various feature types more efficiently.

## File Breakdown
1. Starter_Code_1: The original Colab file used to train the initial model.
2. AlphabetSoupCharity.h5: The trained model from Starter_Code_1.
3.  AlphabetSoupCharity_Optimization_1: Colab file with optimization attempts.
4.  AlphabetSoupCharity_Optimization_1.h5: Trained model from Optimization_1.
5.  AlphabetSoupCharity_Optimization_2: Another optimization attempt in a new Colab file.
6.  AlphabetSoupCharity_Optimization_2.h5: Trained model from Optimization_2.