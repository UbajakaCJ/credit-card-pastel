# Credit Card Fraud Detection for Pastel Dataset

This is a financial dataset from __Pastel__, and the goal of the hackathon is to detect fraudulent transactions. However, the number of fraudulent transactions is very little compared to the normal transactions; the percentage of the normal is __99.79%__, while that of the fraudulent, __0.21%__. It is a skewed dataset.

So, we are battling with an imbalanced dataset. And there are several methods of resolving this:
- Resampling the dataset: Random Undersampling, Oversampling (SMOTE)
- Collecting more data
- Changing the Performance Metric

And we made use of these methods: we undersampled the dataset, and collected the number of normal transactions which is equal to the small number of fraudulent transactions;
and because this dataset is small we collected the fraudulent transaction dataset of the credit card dataset, with an equal number of normal transaction dataset.
Also, instead of using the Accuracy metric, which will incorrectly measure the performance of our model, we changed that to the metric of __Area Under the Receiver Operating Characteristic Curve, ROC_AUC__.

The dataset is highly imbalanced. Therefore, we will not use this dataset as the base of our predictive model, as it will assume that most transactions are not fraud.
We want our model to detect patterns that give signs of fraudulent transactions. 

## Scaling and Sub-Sampling of the Dataset

We will first scale the columns, __Amount__ and __Time__; we normalize them to be on the same scale as other columns.
Also, we will create a sub-sample of the datasets to have an equal amount of Fraudulent and Normal cases. This will help the model understand the patterns relating to whether a transaction is fraudulent or not. We will also, augment the sub-sampled dataset with the sample from the credit card dataset.

A __Sub-Sample__ will be a data frame with a 50/50 ratio of fraudulent and non-fraudulent transactions; using a sub-sampled dataset will prevent:
- __Overfitting:__ Where our model will assume that there is no fraud, and incorrectly classify our dataset
- __Wrong Correlations:__ Where we can have a clear correlation of the features of the datasets and how they influence the prediction of fraud.


### Summary
-  __scaled amount__ and __scaled time__ are the scaled columns in the dataset.
- There are __469__ and __492__ fraudulent cases in both __Pastel__ and __Credit Card__ datasets respectively. So we get corresponding __(469 + 492)__ cases of normal cases in both datasets.
- We concatenate the two datasets to get a new sub-sample of __1922__ cases.

### Random Under-Sampling

Here, we will implement __Random Under Sampling__ which basically consists in creating an equal amount of __Normal__ and __Fraudulent__ cases in the datasets, and removing the excess data cases, in order to have a more balanced dataset and avoid __overfitting__.

__Steps:__
- Bring the __Normal__ transaction datasets to the same amount as __Fraudulent__ transaction sets, this will be equivalent to __(469 + 492)__ cases of normal transactions from both datasets, making a total number of __1922__ cases in the sub-sampled dataset
- After sub-sampling the dataset, we will __reshuffle the dataset__ to see if our models can maintain a certain measure of the performance metric we use, whenever we run our code.

### Classifiers (Under Sampling):

In this section, we will run two types of classifiers and decide which classifier will be more effective in detecting __fraudulent transactions__. 
But before that, we have to split our dataset into training and testing sets and separate the features from the labels.

## Summary

- __Random Forest Classifier__ is a better model among the two models, with a high ROC score of __0.9035__, much better than the __Logistic Regression Classifier__ with ROC score of __0.878__
- __Random Forest Classifier__ shows the best score in both training and testing sets.
- _Random Forests_ are the easiest to train, because they are extremely resilient to hyperparameter choices and require very little preprocessing. They are very fast to train, and should not overfit if you have enough trees. 
