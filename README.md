# Machine-Learning-Portfolio

In-progress Projects
- Spam Filter

Completed Projects
- [Student Score Prediction](#Student-Score-Prediction)

## Student Score Prediction

**Objective**: Developed a machine learning regression model to predict a student's final academic performance (G3 score) to help school advising teams proactively identify at-risk students and allocate intervention resources effectively.

**Dataset**: Utilized a dataset from the UC Irvine Machine Learning Repository.

#### Methodology & Approach:

**Data Preprocessing & Pipelines**: Built robust data pipelines using scikit-learn, including missing value imputation, scaling, and a custom transformer to engineer features (such as aggregating total absences) and dynamically drop features for testing.

**Model Exploration**: Tested and evaluated three regression algorithms: Linear Regression, Linear Support Vector Regression (LinearSVR), and Lasso Regression.

**Dual-Model Strategy**: Specifically investigated the challenge of predicting final grades both with and without preliminary term grades (G1 and G2). This was done to evaluate the feasibility of early intervention before preliminary grades are even recorded.

**Hyperparameter Tuning**: Employed GridSearchCV to fine-tune LinearSVR model parameters for optimal performance. Hyperparameter tuned are C, max_iter, and tol.

**Skills Demonstrated**: Data Cleaning, Feature Engineering, Custom Scikit-Learn Transformers, Pipeline Construction, Cross-Validation, Hyperparameter Tuning, and Business-Focused Model Evaluation.
