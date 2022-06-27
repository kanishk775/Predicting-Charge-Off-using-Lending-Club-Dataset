# Predicting-Charge-Off-using-Lending-Club-Dataset
Predicting who will default on their own using Lending Club Dataset
Introduction
LendingClub is a peer-to-peer lending company. LendingClub enabled borrowers to get unsecured personal loans between $1,000 and $40,000. The standard loan period was three years. Investors were able to search and browse the loan listings on LendingClub website and select loans that they wanted to invest in based on the information supplied about the borrower, amount of loan, loan grade, and loan purpose. Investors made money from the interest on these loans. LendingClub made money by charging borrowers an origination fee and investors a service fee.
Data Description
Our goal for this project is to build a machine learning model to predict if the loan will be charged off or not. We will use only the data available in the dataset and not add any more features as that will be the data available to investors via LendingClub. Such a predictive model could help LendingClub investors make better investment decisions. Our model will be a supervised classification model as it will classify loan seekers into 2 classes, ‘Will fully pay the loan’ and ‘Will default on their loan’.
The raw dataset contains 2.26 million rows and 145 columns. Total size of the dataset is 2.4+ GB. An excel sheet with description of all the labels is attached with this proposal. The predictive column name is loan_status and contains the following values:
Fully Paid 646,902
Current 788,950 Charged Off 168,084
Late (31-120 days) 23,763
In Grace Period 10,474
Late (16-30 days) 5,786
Does not meet the credit policy. Status: Fully Paid 1,988
Does not meet the credit policy. Status: Charged Off 761
Default 70
Research Questions
Few research questions we came across while exploring the data:
1. Which column is our predictive column?
The predictive column name is loan_status.
2. Which parameters in our predictive column are necessary and which can be removed if any We will only use Fully Paid and Charged off. Fully Paid 646902 Charged Off 168084 We will remove the following columns: Current Late (31-120 days) In Grace Period Late (16-30 days) Does not meet the credit policy. Status: Fully Paid Does not meet the credit policy. Status: Charged Off Default
3. Should we classify the data into segments? In how many segments can we classify the data.
We have classified the data into 2 segments: Charged Off and Fully Paid. 79.3% of our data is of customers who fully paid their loan and 20.6% of our data is of customers who defaulted on their loan. We have an imbalanced dataset which is a common practice in the loan dataset as the number of customers, who fully pay back their loan, will be high.
4. How much missing data is acceptable?
We drop columns which contain more than 30% NA values as it is difficult to impute (adding 0 value instead of NA) columns with more than 30% missing/null values. Refer to the chart here.
5. Which all features are irrelevant and which are relevant
Here are the Irrelevant Columns and we drop them for the below reasons:
The column has more than 30% missing values
The column has no correlation with the predictive column or have too many unique values. Example: Title, ID
Here are the Relevant Columns
6. Which relevant features are important and what is the correlation between them?
Attached in the Appendix is the graph for Feature Importance for the Random Forest Model, which is the model with highest Accuracy, Precision and AUC.
7. Which models can be used and how to optimize its hyperparameters?
We used the following models:
Techniques Used
Prediction – Classifiers Dummy Classifier, Logistic Regression (Hyperparameters: max_iterations = 1000 and C = 1), Random Forest (Tweaked max_features = 10), Gradient Boosted Trees (Hyperparameter: learning_rate = 0.01, max_depth = 16), Neural Network (Hyperparameter: hidden_layer_sizes = [50,15], alpha = 0.0001, activation='relu', solver='lbfgs', learning_rate_init=0.001, max_iter=1000), Logistic Regression with cross validation (5 Fold Validation), Random Forest with cross validation (5 Fold Validation)
Evaluation Dummy Classifier, Confusion Matrix, Accuracy, AUC, Precision, Recall, F1 Score
Gradient Boosted Trees: Gradient Boosted Trees and Random Forests are both ensembling methods that perform regression or classification by combining the outputs from individual trees. They both combine many decision trees to reduce the risk of overfitting that each individual tree faces. However, they differ in the way the individual trees are built, and the way the results are combined. Random forests use bagging to build independent decision trees and combine them in parallel. On the other hand, gradient boosted trees use a method called boosting. Boosting combines weak learners (usually decision trees with only one split, called decision stumps) sequentially, so that each new tree corrects the errors of the previous one. We choose the learning rate such that we don’t walk too far in any direction. At the same time, if the learning rate is too low, then the model might take too long to converge to the right answer.
Analysis
LendingClub Loan dataset is a classification problem to predict charge off of loans by borrowers. Our prediction column is loan_status. We are only concerned with Fully Paid and Charged Off (Defaulted). Current means that the loan is ongoing and we don’t know the outcome at the end if the user will default or not. Late means that the user has exceeded the loan duration, but it is not clear if the user will default or not. In Grace period means that the user has been allotted extra time to pay the loan but the outcome is not clear. Does not meet the credit policy means that an exception has been made while extending a loan to them. Therefore, the rest of the values will not help in our analysis and will be dropped. After dropping the values, our dataset contains 80% of Fully paid rows and 20% of Charged Off
rows. We further drop columns which contain more than 30% NA values as it is difficult to impute (adding 0 value instead of NA) columns with more than 30% missing/null values. After dropping such columns, we are left with 92 columns. Since our data has a lot of features, we need to further reduce the features to reduce the model complexity and get a sweet spot where predictive error between train set and test set is low. Since we are making a model to take decisions on new loan applications, we are going to remove the information which will not be available during the initial process of the loan. After removing such columns, we are left with 31 columns. We further analyzed each of the individual columns. Graphs supporting these 31 columns and their correlation with our predictive column are attached in the appendix. The data was split into Train & Test in the proportion of 90th Quantile. We used a simple imputer function with ‘Most Frequent’ strategy to fill all the NA values in our Train and Test dataframe respectively. We then used a dummy classifier to check the score with the features. We are using Confusion Matrix, Accuracy, AUC, Precision, Recall, F1 Score as our markers. Results of the model are in the appendix. We then used Logistic Regression, Random Forest, Gradient Boosted Trees, Neural Network, Logistic Regression with Cross Validation, Random Forest with Cross Validation. Results of the models are in the appendix. Random Forest with Cross Validation yielded the best AUC and Precision Score.
