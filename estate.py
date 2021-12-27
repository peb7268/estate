### Resources ##
#
# Following:
# https://towardsdatascience.com/predicting-house-prices-with-machine-learning-62d5bcd0d68f
# Step by step: https://hub.gke2.mybinder.org/user/john-adeojo-advancedregression-qiez1jcj/notebooks/Kaggle%20Advanced%20House%20Price%20v9.ipynb
#
# 
# Notes:
# * Labels              - in this project are the sale prices of each house
# * Features            - are the independent variables that impact your prediction ( inputs ), in this case sqft, number of bedrooms, number of bathrooms, ect..
# * training data       - is the data that is used to train you model
# * test data           - is the data that is used to test the validity of the predictions from the trained model
# * 
### Comments Key ##
# S - denotes a step
#
# ----------------------------------------------------------------------------------------------------------------------------------
#

# S1. Import libraries fro handling data
# Data / Math imports
import numpy as np
import pandas as pd

# Visualization Imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libs
import time
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Convert training and test data to one hot encoded numeric data

# Create a onehotencoder object that relables columns after transforming
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

# S2. Read in test and train data

# DEFINE FUNCTION TO READ FILES FROM GITHUB REPOS
# Returns a pandas object
def read_file(url):

    """
    Takes GitHub url as an argument,
    pulls CSV file located @ github URL.

    """

    url = url + "?raw=true"
    df = pd.read_csv(url)
    return df


# Test Data - READ FILE FROM GITHUB REPO
url = "https://github.com/john-adeojo/AdvancedRegression/blob/main/test.csv"
Test = read_file(url)

# Training Data - READ ASTRONAUT MISSION FILE FROM GITHUB
url = "https://github.com/john-adeojo/AdvancedRegression/blob/main/train.csv"
Train = read_file(url)
Train.head()

# Outputs Test (1459, 80) Train (1460, 81)
print("Test", Test.shape, "Train", Train.shape)


# S3. Data Wrangling
print("Rows     : ", Train.shape[0])
print("Columns  : ", Train.shape[1])
print("\nFeatures : \n", Train.columns.tolist())
print("\nMissing values :  ", Train.isnull().sum().values.sum())
print("\nUnique values :  \n", Train.nunique())

print("S3. Test", Test.shape, "Train", Train.shape)


# S4. Get a summary of all features ( data variables that influence outcomes, IE. comp sale prices, sq ft, # bedrooms, # bathrooms, ect.. ) 
# in the data and their types
Train.info()

# Remove duplicates from training data
Train.duplicated(subset=None, keep="first")
print("S4. Test", Test.shape, "Train", Train.shape)


# S5. next I'll check for missing values in training data
pd.options.display.min_rows = 115
Train.isnull().sum().sort_values(ascending=False)
print("S5. Missing info - Test", Test.shape, "Train", Train.shape)


# S6. Decide how to handle data holes - Lets get info on the missing values and decide what we will do with them
missing = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "LotFrontage",
    "GarageCond",
    "GarageType",
    "GarageYrBlt",
    "GarageFinish",
    "GarageQual",
    "BsmtExposure",
    "BsmtFinType2",
    "BsmtFinType1",
    "BsmtCond",
    "BsmtQual",
    "MasVnrArea",
    "MasVnrType",
    "Electrical",
]

Train[missing].info()

# Assume poolQC missing values are due to property not having pool (which makes sense for the data )
values = {
    "PoolQC": "No Pool",
    "MiscFeature": "No Feature",
    "Alley": "No alley access",
    "Fence": "No Fence",
    "FireplaceQu": "No Fireplace",
    "GarageCond": "No Garage",
    "GarageType": "No Garage",
    "GarageArea": 0,
    "GarageYrBlt": "None built",
    "GarageFinish": "No Garage",
    "GarageQual": "No Garage",
    "BsmtExposure": "No Basement",
    "BsmtFinType2": "Not Applicable",
    "BsmtFinType1": "Not Applicable",
    "BsmtCond": "No Basement",
    "BsmtQual": "No Basement",
    "MasVnrArea": 0,
    "MasVnrType": "No Veneer",
    "LotFrontage": 0,
}

Train.fillna(value=values, inplace=True)
Test.fillna(value=values, inplace=True)

Train.isna().sum().sort_values(ascending=False)

# for other missing categories we will replace with the mode
features = Train.select_dtypes(include=["object"]).columns

for features in features:
    Train[features].fillna(Train[features].mode()[0], inplace=True)
    Test[features].fillna(Test[features].mode()[0], inplace=True)


# some basement values are set to NANs when they should be zero
Basementmetrics = [
    "BsmtHalfBath",
    "BsmtFullBath",
    "BsmtFinSF1",
    "GarageCars",
    "TotalBsmtSF",
    "BsmtUnfSF",
    "BsmtFinSF2",
]

for Basementmetrics in Basementmetrics:
    Test.loc[(Test[Basementmetrics].isnull()), Basementmetrics] = 0
    Train.loc[(Train[Basementmetrics].isnull()), Basementmetrics] = 0

# mising info for Garage cars should be replaced
Test.loc[(Test.GarageCars.isnull()), "GarageCars"] = 0
Train.loc[(Train.GarageCars.isnull()), "GarageCars"] = 0
print("S6. Test", Test.shape, "Train", Train.shape)


# S7. Normalize Data - Change variable types

# Year built is currently an integer we should treat this as a category for the purpose of this task
Train.YearBuilt = Train.YearBuilt.astype(str)
Test.YearBuilt = Test.YearBuilt.astype(str)

Train.YrSold = Train.YrSold.astype(str)
Test.YrSold = Test.YrSold.astype(str)

Train.GarageYrBlt = Train.GarageYrBlt.astype(str)
Test.GarageYrBlt = Test.GarageYrBlt.astype(str)

Train.YearRemodAdd = Train.YearRemodAdd.astype(str)
Test.YearRemodAdd = Test.YearRemodAdd.astype(str)


# MSSUbCLass, Overallcond & OverallQual: we will decode this to avoid numeric mix-up
MSSUbCLass = {
    20: "1-STORY 1946 & NEWER ALL STYLES",
    30: "1-STORY 1945 & OLDER",
    40: "1-STORY W/FINISHED ATTIC ALL AGES",
    45: "1-1/2 STORY - UNFINISHED ALL AGES",
    50: "1-1/2 STORY FINISHED ALL AGES",
    60: "2-STORY 1946 & NEWER",
    70: "2-STORY 1945 & OLDER",
    75: "2-1/2 STORY ALL AGES",
    80: "SPLIT OR MULTI-LEVEL",
    85: "SPLIT FOYER",
    90: "DUPLEX - ALL STYLES AND AGES",
    120: "1-STORY PUD (Planned Unit Development) - 1946 & NEWER",
    150: "1-1/2 STORY PUD - ALL AGES",
    160: "2-STORY PUD - 1946 & NEWER",
    180: "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
    190: "2 FAMILY CONVERSION - ALL STYLES AND AGES",
}


OverallQualCond = {
    10: "Very Excellent",
    9: "Excellent",
    8: "Very Good",
    7: "Good",
    6: "Above Average",
    5: "Average",
    4: "Below Average",
    3: "Fair",
    2: "Poor",
    1: "Very Poor",
}


Train.replace(
    {
        "OverallQual": OverallQualCond,
        "OverallCond": OverallQualCond,
        "MSSubClass": MSSUbCLass,
    },
    inplace=True,
)
Test.replace(
    {
        "OverallQual": OverallQualCond,
        "OverallCond": OverallQualCond,
        "MSSubClass": MSSUbCLass,
    },
    inplace=True,
)

print("S7. Test", Test.shape, "Train", Train.shape)

# S8. EDA Exploratory Data Analysis
# let's do some descriptive statistics on our data to make sure nothing looks unusual
pd.set_option("display.max_columns", None)
Train.describe()


# S9. EDA Visualization
def showHist(): 
    print("Showing sales price hist...")
    # Sales Price Histogram
    x = Train.SalePrice
    sns.set_style("whitegrid")
    sns.distplot(x)
    
    plt.show()

showHist()

# Meant to minimize Rsq loss - https://www.instapaper.com/read/1469273676 ( minimising squared error )
# Basically  converts to 10's instead of 000's to make the price spread more narrow and the results more focused
# This needs to run bc it sets the Training data SalePrice_log property
def showFlattnedHist():
    print("Showing flattened sales price hist...")
    Train["SalePrice_log"] = np.log(Train.SalePrice)
    x = Train.SalePrice_log
    sns.distplot(x)
    plt.show()

showFlattnedHist()

# S10. Correlations in dataset via heatmap - Lets explore the correlations in our data set
def showHeatmap():
    print("showing heatmap...")
    plt.figure(figsize=(10, 10))
    sns.heatmap(Train.corr())
    plt.savefig("pearsonsmep.png")
    plt.show()

# showHeatmap()

# S11. Transform the data and prep it for ML
# Wrapper for one hot encoder to allow labelling of encoded variables

class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X)
        d_out = pd.DataFrame(
            sparse_matrix.toarray(), columns=new_columns, index=X.index
        )
        return d_out

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f"{column}_<{self.categories_[i][j]}>")
                j += 1
        return new_columns


# S12. Explore the data a bit more
# We will plot some joint histogram and scatter grphs to look at correlated features in more detail
y = Train.SalePrice

def plotFeaturesForPrice():
    features = [
        "MasVnrArea",
        "BsmtFinSF1",
        "TotalBsmtSF",
        "1stFlrSF",
        "GrLivArea",
        "FullBath",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageCars",
        "GarageArea",
        "LotArea",
        "LotFrontage",
    ]

    for features in features:
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 10))
        x = Train[features]
        sns.jointplot(x=x, y=y, data=Train)
        
        plt.show()

plotFeaturesForPrice()

# S13.
# Define funtion to encode categorrical variables with and rejoin to initial data
# Converts categorical data into numbers and puts it back in the original data set
def transform(Train, df):

    # isolate categorical features
    cat_columns = df.select_dtypes(include=["object"]).columns
    cat_df = df[cat_columns]

    # isolate the numeric features
    numeric_df = df.select_dtypes(include=np.number)

    # initialise one hot encoder object spcify handle unknown and auto options to keep test and train same size
    ohe = OneHotEncoder(categories="auto", handle_unknown="ignore")
    # Fit the endcoder to training data
    ohe.fit(Train[cat_columns])

    # transform input data
    df_processed = ohe.transform(cat_df)

    # concatinate numeric features from orginal tables with encoded features
    df_processed_full = pd.concat([df_processed, numeric_df], axis=1)

    return df_processed_full


# Get transformed Training data ( Training data with categorical data mapped to numbers )
Train_encoded = transform(Train, Train)

# Get transformed test data ( test data with categorical data mapped to numbers )
Test_encoded = transform(Train, Test)

# Check data sets are same width minus the two labels in Train
print("Test", Test_encoded.shape, "Train", Train_encoded.shape)


# S12. Begin ML with SciKit and the Random Forrest Model
# Model 1: Ranbdom Forest Rgressor

# Split training data into features (x_train) and labels (Y_train)
x_train = Train_encoded.drop(columns=["Id", "SalePrice", "SalePrice_log"])
Y_train = Train_encoded.SalePrice_log


# Initialise Gridsearch CV with 5 fold cross-validation and root mse for socirn
def model_pipeline(model, param_grid, scoring):
    print("Creating GridSearchCV...")
    Tuned_Model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=5)
    print("GridSearchCV Created...")

    # Fit model & Time the process for training the model
    print("running model")
    start_time = time.process_time()

    print("Fitting Model...")
    Tuned_Model.fit(x_train, Y_train)

    # End of fit time
    print(time.process_time() - start_time, "Seconds")
    print("finished running model")

    return Tuned_Model


# Generate results of best run
def plot_mean_scores(Tuned_Model, col, x, y, name):
    Results = pd.DataFrame(Tuned_Model.cv_results_)
    Results_Best = Results.loc[Results.rank_test_score == 1]

    # Initialize a grid of plots
    sns.set(font_scale=1.5)
    sns.set_style("darkgrid")

    col = col
    for col in col:
        grid = sns.FacetGrid(
            Results,
            col=col,
            hue="rank_test_score",
            palette="tab20c",
            legend_out=False,
            col_wrap=5,
            height=5,
        )

        # Draw a horizontal line to show the starting point
        grid.map(plt.axhline, y=0, ls=":", c=".5")

        # Draw marker on plot and decide what parameters to plot
        grid.map(sns.scatterplot, x, y, marker="o", s=200)

        # Adjust the arrangement of the plots
        grid.fig.tight_layout(w_pad=1)

        # Add legend to gird
        grid.add_legend()

    grid.savefig(name)
    
    plt.show()

# Modelling - fitting the data to the model

# Build a decision tree
# Set paramters for Grid Search CV
param_grid = {
    "max_depth": [5, 10, 30],
    "max_features": [0.1, 0.3, 0.7],
}
model = DecisionTreeRegressor(criterion='squared_error', random_state=0)

Tuned_Model = model_pipeline(model, param_grid, "neg_root_mean_squared_error")

plot_mean_scores(
    Tuned_Model,
    ["param_max_depth"],
    "param_max_features",
    "mean_test_score",
    "DecisionTree",
)


# My Code...
print("Predicting next price...")

# Start the script with python3 -m pdb estate.py
# When it stops use the continue (c) keyword. It will go until the breakpoint call
# breakpoint()

# ValueError: X has 1 features, but DecisionTreeRegressor is expecting 606 features as input.
# Test_encoded [1459 rows x 607 columns]
# type(Test_encoded) # class 'pandas.core.frame.DataFrame'

# Tuned_Model.predict(Test_encoded)
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.predict
