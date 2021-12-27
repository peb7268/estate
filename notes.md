
## TODO:
1. Notate python data types
2. Notate mean square regression loss

# Definitions
- Labels: categories you assign to your data to classify your model so model will know how to classify. Spam or not spam, cat or not cat. Used more in supervised learning?
- supervised learning: uses examples and labels to find patterns in data
- Hyperparameters are paramters that you can tweak that deal witht the way the modeling itself works. ( learning_rate, max_features, ect..)
- Model Bias — Models that underfit the training data leading to poor predictive capacity on unseen data. Generally, the simpler the model the higher the bias.
- Model Variance — Models that overfit the training data leading to poor predictive capacity on unseen data. Generally, the more complexity in the model the higher the variance.
- Decision Tree — A tree algorithm used in machine learning to find patterns in data by learning decision rules.
- Random Forest — A type of bagging method that plays on ‘the wisdom of crowds’ effect. It uses multiple independent decision trees 
  in parallel to learn from data and aggregates their predictions for an outcome.
- Gradient Boosting Machines — A type of boosting method that uses a combination of decision tree in series. Each tree is used to 
  predict and correct the errors by the preceding tree additively.

- Histogram: Bar chart


# Notables
* Complexity can be thought of as the number of features in the model.
* Model variance and model bias have an inverse relationship leading to a trade-off.


# Project specific questions
    - not getting the output breakdown like I see in the example for S8?
    - What is the General ML pipeline process?
        1. Data ingestion: in this project currently from a github url
        2. Data cleaning: 
        3. Exploratory Data Analysis (EDA): 
        4. Feature Engineering: 
        5. Machine Learning:

# General Questions
- Classifications of ML? 
    * supervised learning
    * unsupervised learning
    * reinforced learning

- What is best practices for python dev?
    * virtualenv?
    * pip3 vs pip python3 vs python ect..

- Which algorythims are best for which types of learning? 
    * when to use linear regression?

- ( matplotlib.pyplot ) plt only seems to be able to show 1 chart at a time and have to close it before the next one will render?


# Project Gameplan
1. Understand ML ingestion of data and how the process works
2. Write a mock csv based on the data fields I want to use
3. Feed test and training data
4. once working change it out for simplyreits or MLS
