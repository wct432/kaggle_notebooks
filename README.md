This repo is used to store and share code I have completed for Kaggle competitions. Currently I have only have my housing price 
competition hosted on GitHub so I am displaying its ReadMe here for now.

- [1. Introduction](#1-introduction)
- [2. Exploration, Visualization, and Feature Engineering](#2-exploration-visualization-and-feature-engineering)
  - [1. Exploratory Analysis](#1-exploratory-analysis)
    - [1. Preview Dataset](#1-preview-dataset)
    - [2. View Standard Deviation, Mean, Percentiles for Numerical Features](#2-view-standard-deviation-mean-percentiles-for-numerical-features)
    - [3. Find Columns with Null Values and their Sums](#3-find-columns-with-null-values-and-their-sums)
    - [4. Determine Different Datatypes](#4-determine-different-datatypes)
  - [2. Visualizations and Processing](#2-visualizations-and-processing)
    - [1. Mean Housing Price per Neighborhood](#1-mean-housing-price-per-neighborhood)
    - [2. Feature/Target Correlation](#2-featuretarget-correlation)
    - [3. Feature Correlation](#3-feature-correlation)
    - [4. Distribution of Features](#4-distribution-of-features)
    - [5. Feature Engineering](#5-feature-engineering)
    - [6. Saving Prepared Dataset](#6-saving-prepared-dataset)
- [3. Preprocessing and Preparing Data with Sklearn Pipeline](#3-preprocessing-and-preparing-data-with-sklearn-pipeline)
- [4. Random Forest Model and Hyperparemter Tuning](#4-random-forest-model-and-hyperparemter-tuning)
    - [1. Define Base Model](#1-define-base-model)
    - [2. Randomized Cross Validated Search](#2-randomized-cross-validated-search)
    - [3. Cross Validated Grid Search](#3-cross-validated-grid-search)
    - [4. Save Random Forest with Best Hyperparemters](#4-save-random-forest-with-best-hyperparemters)
- [5. XGBoost Model](#5-xgboost-model)
    - [1. Create Train and Validation Sets](#1-create-train-and-validation-sets)
    - [2. Define Optimization Function](#2-define-optimization-function)
    - [3. Define Optuna Objective](#3-define-optuna-objective)
    - [3. Create Study and Optimize](#3-create-study-and-optimize)
    - [4. View Optuna Results](#4-view-optuna-results)
    - [5. Saving Best Model](#5-saving-best-model)
- [6. CatBoost Model](#6-catboost-model)
    - [1. Create CatBoost Model and Optuna Objective](#1-create-catboost-model-and-optuna-objective)
    - [2. Create Optuna Study and Optimize](#2-create-optuna-study-and-optimize)
    - [3. View Optuna Results](#3-view-optuna-results)
    - [4. Saving Best Model](#4-saving-best-model)
- [7. Create and Fit Ensemble Model](#7-create-and-fit-ensemble-model)
    - [1. Create Ensemble Model](#1-create-ensemble-model)
    - [2. Fit Ensemble Model](#2-fit-ensemble-model)
    - [3. Create Final Predictions and Save Ensemble Model](#3-create-final-predictions-and-save-ensemble-model)

# 1. Introduction
The goal of this project is housing price prediction in Kaggle's Housing Prices Comeptition. 

The goal of the competition is predicting the price of residential homes in Ames, Iowa 
and is scored using the RMSE. The dataset contains 79 features. 

I approached the problem using an ensemble model combining random forest, XGBoost, CatBoost and linear regression models.

The hyperparameters of the decision tree were tuned using a cross-validated random search and grid search,
and the XGBoost and CatBoost models were tuned using Optuna. 




# 2. Exploration, Visualization, and Feature Engineering

## 1. Exploratory Analysis
First we begin by gathering some basic information about our dataset, like its shape, the mean and standard deviation for 
each column, and other information about the data like the number of n/a values. We want to get an idea for how the data feels
and how the data is distributed in different columns. It's also important to determine what data types we have in our data set, 
we will have to deal with features that are categorical by one hot encoding them. 

After getting an overview of our dataset we wil create visualizations to help us learn more about the relationships between the features, 
the target (Sale Price), where the most expensive homes are located, and other information.

### 1. Preview Dataset
``` python
training_set.head()
```
|   Id |   MSSubClass | MSZoning   |   LotFrontage |   LotArea | Street   |   Alley | LotShape   | LandContour   | Utilities   | LotConfig   | LandSlope   | Neighborhood   | Condition1   | Condition2   | BldgType   | HouseStyle   |   OverallQual |   OverallCond |   YearBuilt |   YearRemodAdd | RoofStyle   | RoofMatl   | Exterior1st   | Exterior2nd   | MasVnrType   |   MasVnrArea | ExterQual   | ExterCond   | Foundation   | BsmtQual   | BsmtCond   | BsmtExposure   | BsmtFinType1   |   BsmtFinSF1 | BsmtFinType2   |   BsmtFinSF2 |   BsmtUnfSF |   TotalBsmtSF | Heating   | HeatingQC   | CentralAir   | Electrical   |   1stFlrSF |   2ndFlrSF |   LowQualFinSF |   GrLivArea |   BsmtFullBath |   BsmtHalfBath |   FullBath |   HalfBath |   BedroomAbvGr |   KitchenAbvGr | KitchenQual   |   TotRmsAbvGrd | Functional   |   Fireplaces | FireplaceQu   | GarageType   |   GarageYrBlt | GarageFinish   |   GarageCars |   GarageArea | GarageQual   | GarageCond   | PavedDrive   |   WoodDeckSF |   OpenPorchSF |   EnclosedPorch |   3SsnPorch |   ScreenPorch |   PoolArea |   PoolQC | Fence   | MiscFeature   |   MiscVal |   MoSold |   YrSold | SaleType   | SaleCondition   |   SalePrice |
|-----:|-------------:|:-----------|--------------:|----------:|:---------|--------:|:-----------|:--------------|:------------|:------------|:------------|:---------------|:-------------|:-------------|:-----------|:-------------|--------------:|--------------:|------------:|---------------:|:------------|:-----------|:--------------|:--------------|:-------------|-------------:|:------------|:------------|:-------------|:-----------|:-----------|:---------------|:---------------|-------------:|:---------------|-------------:|------------:|--------------:|:----------|:------------|:-------------|:-------------|-----------:|-----------:|---------------:|------------:|---------------:|---------------:|-----------:|-----------:|---------------:|---------------:|:--------------|---------------:|:-------------|-------------:|:--------------|:-------------|--------------:|:---------------|-------------:|-------------:|:-------------|:-------------|:-------------|-------------:|--------------:|----------------:|------------:|--------------:|-----------:|---------:|:--------|:--------------|----------:|---------:|---------:|:-----------|:----------------|------------:|
|    1 |           60 | RL         |            65 |      8450 | Pave     |     nan | Reg        | Lvl           | AllPub      | Inside      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        2003 |           2003 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          196 | Gd          | TA          | PConc        | Gd         | TA         | No             | GLQ            |          706 | Unf            |            0 |         150 |           856 | GasA      | Ex          | Y            | SBrkr        |        856 |        854 |              0 |        1710 |              1 |              0 |          2 |          1 |              3 |              1 | Gd            |              8 | Typ          |            0 | nan           | Attchd       |          2003 | RFn            |            2 |          548 | TA           | TA           | Y            |            0 |            61 |               0 |           0 |             0 |          0 |      nan | nan     | nan           |         0 |        2 |     2008 | WD         | Normal          |      208500 |
|    2 |           20 | RL         |            80 |      9600 | Pave     |     nan | Reg        | Lvl           | AllPub      | FR2         | Gtl         | Veenker        | Feedr        | Norm         | 1Fam       | 1Story       |             6 |             8 |        1976 |           1976 | Gable       | CompShg    | MetalSd       | MetalSd       | None         |            0 | TA          | TA          | CBlock       | Gd         | TA         | Gd             | ALQ            |          978 | Unf            |            0 |         284 |          1262 | GasA      | Ex          | Y            | SBrkr        |       1262 |          0 |              0 |        1262 |              0 |              1 |          2 |          0 |              3 |              1 | TA            |              6 | Typ          |            1 | TA            | Attchd       |          1976 | RFn            |            2 |          460 | TA           | TA           | Y            |          298 |             0 |               0 |           0 |             0 |          0 |      nan | nan     | nan           |         0 |        5 |     2007 | WD         | Normal          |      181500 |
|    3 |           60 | RL         |            68 |     11250 | Pave     |     nan | IR1        | Lvl           | AllPub      | Inside      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        2001 |           2002 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          162 | Gd          | TA          | PConc        | Gd         | TA         | Mn             | GLQ            |          486 | Unf            |            0 |         434 |           920 | GasA      | Ex          | Y            | SBrkr        |        920 |        866 |              0 |        1786 |              1 |              0 |          2 |          1 |              3 |              1 | Gd            |              6 | Typ          |            1 | TA            | Attchd       |          2001 | RFn            |            2 |          608 | TA           | TA           | Y            |            0 |            42 |               0 |           0 |             0 |          0 |      nan | nan     | nan           |         0 |        9 |     2008 | WD         | Normal          |      223500 |
|    4 |           70 | RL         |            60 |      9550 | Pave     |     nan | IR1        | Lvl           | AllPub      | Corner      | Gtl         | Crawfor        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        1915 |           1970 | Gable       | CompShg    | Wd Sdng       | Wd Shng       | None         |            0 | TA          | TA          | BrkTil       | TA         | Gd         | No             | ALQ            |          216 | Unf            |            0 |         540 |           756 | GasA      | Gd          | Y            | SBrkr        |        961 |        756 |              0 |        1717 |              1 |              0 |          1 |          0 |              3 |              1 | Gd            |              7 | Typ          |            1 | Gd            | Detchd       |          1998 | Unf            |            3 |          642 | TA           | TA           | Y            |            0 |            35 |             272 |           0 |             0 |          0 |      nan | nan     | nan           |         0 |        2 |     2006 | WD         | Abnorml         |      140000 |
|    5 |           60 | RL         |            84 |     14260 | Pave     |     nan | IR1        | Lvl           | AllPub      | FR2         | Gtl         | NoRidge        | Norm         | Norm         | 1Fam       | 2Story       |             8 |             5 |        2000 |           2000 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          350 | Gd          | TA          | PConc        | Gd         | TA         | Av             | GLQ            |          655 | Unf            |            0 |         490 |          1145 | GasA      | Ex          | Y            | SBrkr        |       1145 |       1053 |              0 |        2198 |              1 |              0 |          2 |          1 |              4 |              1 | Gd            |              9 | Typ          |            1 | TA            | Attchd       |          2000 | RFn            |            3 |          836 | TA           | TA           | Y            |          192 |            84 |               0 |           0 |             0 |          0 |      nan | nan     | nan           |         0 |       12 |     2008 | WD         | Normal          |      250000 |
|    6 |           50 | RL         |            85 |     14115 | Pave     |     nan | IR1        | Lvl           | AllPub      | Inside      | Gtl         | Mitchel        | Norm         | Norm         | 1Fam       | 1.5Fin       |             5 |             5 |        1993 |           1995 | Gable       | CompShg    | VinylSd       | VinylSd       | None         |            0 | TA          | TA          | Wood         | Gd         | TA         | No             | GLQ            |          732 | Unf            |            0 |          64 |           796 | GasA      | Ex          | Y            | SBrkr        |        796 |        566 |              0 |        1362 |              1 |              0 |          1 |          1 |              1 |              1 | TA            |              5 | Typ          |            0 | nan           | Attchd       |          1993 | Unf            |            2 |          480 | TA           | TA           | Y            |           40 |            30 |               0 |         320 |             0 |          0 |      nan | MnPrv   | Shed          |       700 |       10 |     2009 | WD         | Normal          |      143000 |
|    7 |           20 | RL         |            75 |     10084 | Pave     |     nan | Reg        | Lvl           | AllPub      | Inside      | Gtl         | Somerst        | Norm         | Norm         | 1Fam       | 1Story       |             8 |             5 |        2004 |           2005 | Gable       | CompShg    | VinylSd       | VinylSd       | Stone        |          186 | Gd          | TA          | PConc        | Ex         | TA         | Av             | GLQ            |         1369 | Unf            |            0 |         317 |          1686 | GasA      | Ex          | Y            | SBrkr        |       1694 |          0 |              0 |        1694 |              1 |              0 |          2 |          0 |              3 |              1 | Gd            |              7 | Typ          |            1 | Gd            | Attchd       |          2004 | RFn            |            2 |          636 | TA           | TA           | Y            |          255 |            57 |               0 |           0 |             0 |          0 |      nan | nan     | nan           |         0 |        8 |     2007 | WD         | Normal          |      307000 |
|    8 |           60 | RL         |           nan |     10382 | Pave     |     nan | IR1        | Lvl           | AllPub      | Corner      | Gtl         | NWAmes         | PosN         | Norm         | 1Fam       | 2Story       |             7 |             6 |        1973 |           1973 | Gable       | CompShg    | HdBoard       | HdBoard       | Stone        |          240 | TA          | TA          | CBlock       | Gd         | TA         | Mn             | ALQ            |          859 | BLQ            |           32 |         216 |          1107 | GasA      | Ex          | Y            | SBrkr        |       1107 |        983 |              0 |        2090 |              1 |              0 |          2 |          1 |              3 |              1 | TA            |              7 | Typ          |            2 | TA            | Attchd       |          1973 | RFn            |            2 |          484 | TA           | TA           | Y            |          235 |           204 |             228 |           0 |             0 |          0 |      nan | nan     | Shed          |       350 |       11 |     2009 | WD         | Normal          |      200000 |
|    9 |           50 | RM         |            51 |      6120 | Pave     |     nan | Reg        | Lvl           | AllPub      | Inside      | Gtl         | OldTown        | Artery       | Norm         | 1Fam       | 1.5Fin       |             7 |             5 |        1931 |           1950 | Gable       | CompShg    | BrkFace       | Wd Shng       | None         |            0 | TA          | TA          | BrkTil       | TA         | TA         | No             | Unf            |            0 | Unf            |            0 |         952 |           952 | GasA      | Gd          | Y            | FuseF        |       1022 |        752 |              0 |        1774 |              0 |              0 |          2 |          0 |              2 |              2 | TA            |              8 | Min1         |            2 | TA            | Detchd       |          1931 | Unf            |            2 |          468 | Fa           | TA           | Y            |           90 |             0 |             205 |           0 |             0 |          0 |      nan | nan     | nan           |         0 |        4 |     2008 | WD         | Abnorml         |      129900 |
|   10 |          190 | RL         |            50 |      7420 | Pave     |     nan | Reg        | Lvl           | AllPub      | Corner      | Gtl         | BrkSide        | Artery       | Artery       | 2fmCon     | 1.5Unf       |             5 |             6 |        1939 |           1950 | Gable       | CompShg    | MetalSd       | MetalSd       | None         |            0 | TA          | TA          | BrkTil       | TA         | TA         | No             | GLQ            |          851 | Unf            |            0 |         140 |           991 | GasA      | Ex          | Y            | SBrkr        |       1077 |          0 |              0 |        1077 |              1 |              0 |          1 |          0 |              2 |              2 | TA            |              5 | Typ          |            2 | TA            | Attchd       |          1939 | RFn            |            1 |          205 | Gd           | TA           | Y            |            0 |             4 |               0 |           0 |             0 |          0 |      nan | nan     | nan           |         0 |        1 |     2008 | WD         | Normal          |      118000 |

### 2. View Standard Deviation, Mean, Percentiles for Numerical Features
``` python
training_set.describe()
```
|       |   MSSubClass |   LotFrontage |   LotArea |   OverallQual |   OverallCond |   YearBuilt |   YearRemodAdd |   MasVnrArea |   BsmtFinSF1 |   BsmtFinSF2 |   BsmtUnfSF |   TotalBsmtSF |   1stFlrSF |   2ndFlrSF |   LowQualFinSF |   GrLivArea |   BsmtFullBath |   BsmtHalfBath |    FullBath |    HalfBath |   BedroomAbvGr |   KitchenAbvGr |   TotRmsAbvGrd |   Fireplaces |   GarageYrBlt |   GarageCars |   GarageArea |   WoodDeckSF |   OpenPorchSF |   EnclosedPorch |   3SsnPorch |   ScreenPorch |   PoolArea |   MiscVal |     MoSold |    YrSold |   SalePrice |
|:------|-------------:|--------------:|----------:|--------------:|--------------:|------------:|---------------:|-------------:|-------------:|-------------:|------------:|--------------:|-----------:|-----------:|---------------:|------------:|---------------:|---------------:|------------:|------------:|---------------:|---------------:|---------------:|-------------:|--------------:|-------------:|-------------:|-------------:|--------------:|----------------:|------------:|--------------:|-----------:|----------:|-----------:|----------:|------------:|
| count |    1460      |     1201      |   1460    |    1460       |    1460       |   1460      |      1460      |     1452     |     1460     |    1460      |    1460     |      1460     |   1460     |   1460     |     1460       |     1460    |    1460        |   1460         | 1460        | 1460        |    1460        |    1460        |     1460       |  1460        |     1379      |  1460        |     1460     |    1460      |     1460      |       1460      |  1460       |     1460      |  1460      |  1460     | 1460       | 1460      |      1460   |
| mean  |      56.8973 |       70.05   |  10516.8  |       6.09932 |       5.57534 |   1971.27   |      1984.87   |      103.685 |      443.64  |      46.5493 |     567.24  |      1057.43  |   1162.63  |    346.992 |        5.84452 |     1515.46 |       0.425342 |      0.0575342 |    1.56507  |    0.382877 |       2.86644  |       1.04658  |        6.51781 |     0.613014 |     1978.51   |     1.76712  |      472.98  |      94.2445 |       46.6603 |         21.9541 |     3.40959 |       15.061  |     2.7589 |    43.489 |    6.32192 | 2007.82   |    180921   |
| std   |      42.3006 |       24.2848 |   9981.26 |       1.383   |       1.1128  |     30.2029 |        20.6454 |      181.066 |      456.098 |     161.319  |     441.867 |       438.705 |    386.588 |    436.528 |       48.6231  |      525.48 |       0.518911 |      0.238753  |    0.550916 |    0.502885 |       0.815778 |       0.220338 |        1.62539 |     0.644666 |       24.6897 |     0.747315 |      213.805 |     125.339  |       66.256  |         61.1191 |    29.3173  |       55.7574 |    40.1773 |   496.123 |    2.70363 |    1.3281 |     79442.5 |
| min   |      20      |       21      |   1300    |       1       |       1       |   1872      |      1950      |        0     |        0     |       0      |       0     |         0     |    334     |      0     |        0       |      334    |       0        |      0         |    0        |    0        |       0        |       0        |        2       |     0        |     1900      |     0        |        0     |       0      |        0      |          0      |     0       |        0      |     0      |     0     |    1       | 2006      |     34900   |
| 25%   |      20      |       59      |   7553.5  |       5       |       5       |   1954      |      1967      |        0     |        0     |       0      |     223     |       795.75  |    882     |      0     |        0       |     1129.5  |       0        |      0         |    1        |    0        |       2        |       1        |        5       |     0        |     1961      |     1        |      334.5   |       0      |        0      |          0      |     0       |        0      |     0      |     0     |    5       | 2007      |    129975   |
| 50%   |      50      |       69      |   9478.5  |       6       |       5       |   1973      |      1994      |        0     |      383.5   |       0      |     477.5   |       991.5   |   1087     |      0     |        0       |     1464    |       0        |      0         |    2        |    0        |       3        |       1        |        6       |     1        |     1980      |     2        |      480     |       0      |       25      |          0      |     0       |        0      |     0      |     0     |    6       | 2008      |    163000   |
| 75%   |      70      |       80      |  11601.5  |       7       |       6       |   2000      |      2004      |      166     |      712.25  |       0      |     808     |      1298.25  |   1391.25  |    728     |        0       |     1776.75 |       1        |      0         |    2        |    1        |       3        |       1        |        7       |     1        |     2002      |     2        |      576     |     168      |       68      |          0      |     0       |        0      |     0      |     0     |    8       | 2009      |    214000   |
| max   |     190      |      313      | 215245    |      10       |       9       |   2010      |      2010      |     1600     |     5644     |    1474      |    2336     |      6110     |   4692     |   2065     |      572       |     5642    |       3        |      2         |    3        |    2        |       8        |       3        |       14       |     3        |     2010      |     4        |     1418     |     857      |      547      |        552      |   508       |      480      |   738      | 15500     |   12       | 2010      |    755000   |

### 3. Find Columns with Null Values and their Sums

``` python
#select columns that contain null values and the sum of the null's as a tuple
na_cols = [(i,training_set[i].isna().sum()) for i in training_set.columns if training_set[i].isna().any()]
print(na_cols)
```

*[('LotFrontage', 259), ('Alley', 1369), ('MasVnrType', 8), ('MasVnrArea', 8), ('BsmtQual', 37), ('BsmtCond', 37), ('BsmtExposure', 38), ('BsmtFinType1', 37), ('BsmtFinType2', 38), ('Electrical', 1),*   
*[('FireplaceQu', 690), ('GarageType', 81), ('GarageYrBlt', 81), ('GarageFinish', 81), ('GarageQual', 81), ('GarageCond', 81), ('PoolQC', 1453), ('Fence', 1179), ('MiscFeature', 1406)]*

### 4. Determine Different Datatypes
``` python
#display unique datatypes 
print(training_set.dtypes.unique())
```
*[dtype('int64') dtype('O') dtype('float64')]*

## 2. Visualizations and Processing

### 1. Mean Housing Price per Neighborhood 
<details>
<summary> Click to display code! </summary>

<p>

``` python
#create a dataframe with the average housing price per neighborhood
mean_neighborhood_price = training_set.groupby('Neighborhood').SalePrice.median() \
                        .sort_values(ascending = False).reset_index()


sns.set(font_scale=1.5) #font size
fig, ax = plt.subplots(figsize=(15,7))                      #determine figure size 
ax = sns.barplot(data = mean_neighborhood_price,            #setting data for plot to mean_neighborhood df 
                x=mean_neighborhood_price.Neighborhood,     #setting x to neighborhod
                y=mean_neighborhood_price.SalePrice,        #setting y to Sale Price
                palette="Spectral")                         #us spectral color pallete
ax.set_xlabel('Neighborhood')
ax.set_ylabel('Housing Price', fontsize = 20)
ax.tick_params(axis='x', labelrotation= 40, labelsize=15)   #rotating x labels so they are more legible
```

</p>
</details>

![Median Sale Price by Neighborhood](/../images/images/neighborhood_median_prices.png?raw=true)


### 2. Feature/Target Correlation
Heatmap that showing correlation each feature has with between features and Housing Price
<details>
<summary> Click to display code! </summary>

<p>

``` python
#create a correlation heatmap to show how each feature correlates to our target
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(training_set.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False), \
                        vmin=-1, vmax=1, annot=True, cmap='rocket')

heatmap.set_title('Features Correlating with Sales Price', fontdict={'fontsize':18}, pad=16)
```

</p>
</details>

![Features and Target Correlation](/../images/images/feature_target_correlation.png?raw=true)

### 3. Feature Correlation
Heatmap that shows correlation between different features in the dataset. 
<details>
<summary> Click to display code! </summary>

<p>

``` python
#create a correlation heatmap to show how each feature correlates to our target
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(training_set.corr}, pad=16)
```

</p>
</details>

![Feature Correlations](/../images/images/feature_correlations.png?raw=true)


### 4. Distribution of Features
Create histograms for each feature, as well as the target to visualize the distribution of the data. 
<details>
<summary> Click to display code! </summary>

<p>

``` python
#plot histograms for each feature
_ = training_set.hist(bins = 50, figsize = (25,20))
```

</p>
</details>

![Dataset Histograms](/../images/images/dataset_histograms.png?raw=true)


### 5. Feature Engineering
Next we are going to create  new features, we will create baths per square foot,  
as well as a number of binary columns that represent if the homes have a certain attribute,  
for example if they have a 2nd story, have a fireplace, etc. We will also binarize a few features  
in place using the same technique. 

We will also replace any null and infinite values with 0.  

We will deal with normalizing the disitribution of our data in the next section using our Sklearn pipeline.

``` python
#create new features
for sets in training_set, testing_set:
    sets['baths_per_sf'] = sets['1stFlrSF'] / (sets['FullBath'] + sets['HalfBath'])
    sets['remodeled'] = sets['BsmtFinSF2'].apply(lambda x: 1 if x != 0 else x)
    sets['has_fireplace'] = sets['Fireplaces'].apply(lambda x: 1 if x != 0 else x)
    sets['has_porch'] = sets['OpenPorchSF'].apply(lambda x: 1 if x != 0 else x)
    sets['has_2nd_story'] = sets['2ndFlrSF'].apply(lambda x: 1 if x != 0 else x)
    sets['has_garage'] = sets['GarageArea'].apply(lambda x: 1 if x != 0 else x)

    #binarize these features in place 
    for col in ['BsmtFinSF2','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea']:
        sets[col] = sets[col].apply(lambda x: 1 if x != 0 else x)
        
    #replace inf and na values with 0
    sets.replace([np.inf, -np.inf], 0, inplace=True)
    sets.fillna(0, inplace=True)

#view correlation with the newly added and modified features included
training_set.corrwith(training_set['SalePrice']).sort_values(ascending=False)
```



<details>
<summary> Click to View Output (Correlation of Features with Target after Additions) </summary>

<p>

SalePrice        1.000000  
OverallQual      0.790982  
GrLivArea        0.708624  
GarageCars       0.640409  
GarageArea       0.623431  
TotalBsmtSF      0.613581  
1stFlrSF         0.605852  
FullBath         0.560664  
TotRmsAbvGrd     0.533723  
YearBuilt        0.522897  
YearRemodAdd     0.507101  
MasVnrArea       0.472614  
has_fireplace    0.471908  
Fireplaces       0.466929  
has_porch        0.412959  
BsmtFinSF1       0.386420  
WoodDeckSF       0.324413  
2ndFlrSF         0.319334  
OpenPorchSF      0.315856  
HalfBath         0.284108  
LotArea          0.263843  
GarageYrBlt      0.261366  
has_garage       0.236832  
BsmtFullBath     0.227122  
BsmtUnfSF        0.214479  
LotFrontage      0.209624  
BedroomAbvGr     0.168213  
has_2nd_story    0.137656  
PoolArea         0.093708  
ScreenPorch      0.087143  
MoSold           0.046432  
3SsnPorch        0.046015  
BsmtHalfBath    -0.016844  
MiscVal         -0.021190  
LowQualFinSF    -0.025606  
YrSold          -0.028923  
BsmtFinSF2      -0.052965  
remodeled       -0.052965  
OverallCond     -0.077856  
MSSubClass      -0.084284  
baths_per_sf    -0.125778  
KitchenAbvGr    -0.135907  
EnclosedPorch   -0.183374  
dtype: float64  

</p>
</details>


### 6. Saving Prepared Dataset

``` python
processed_path = data_path + "processed/"               #path for proessed data
y = training_set['SalePrice']                           #set training target 
training_set.drop(['SalePrice'], axis=1, inplace=True)  #drop target from df
X = training_set                                        #set training feature
X.to_csv(processed_path + 'X.csv')                      #save features as csv
np.save(processed_path + 'y.npy',y)                     #save target as npy
testing_set.to_csv(processed_path + 'testing_set.csv',  #save testing set
                    index = 'Id')
```


# 3. Preprocessing and Preparing Data with Sklearn Pipeline
Now we are working in the rand_forest notebook where we initially preprocess the data  
using an Scikit-Learn pipeline and then tune a random forest model using cross validated grid  
and random searches. 


First we begin by splitting our data into train and testing sets. 

``` python 
#split train and test sets, validation set not needed as we will be using CV
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8,
                                                      test_size=0.2, random_state = 42)
```                                   

Next we will define our preprocessing pipeline using Scikit-Learn.  
This pipeline will be responsible for normalizing the distribution of our features,  
filling any n/a values as a precautionary (should have been addressed in prior steps),   
and it will also one hot encode the categorical features. We will save the preprocessor  
in our /models/ directory so that we can easily use it later.    

Normalizing the distribution of features is not necessary for tree based algorithms   
like random forest and XGBoost because they split each node in the tree using one feature  
at at time, and as a result the scale of other features is irrelevant. However, I found I   
had slightly better results after using the quantile transformer to transform each feature   
into a normal distribution.  

``` python
#NUMERICAL PIPELINE
num_cols = X.select_dtypes(exclude="object").columns                            #select numerical columns in df
num_transformer = Pipeline(steps=[                                              #define pipeine for numerical columns
    ('imputer', SimpleImputer(strategy='constant')),                            #fill any missing values with 0
    ('quantile_transformer', QuantileTransformer(output_distribution='normal',  #transform features into following a normal distribution
                                                 n_quantiles=700,               #set number of quantiles to computer for distribution to 700
                                                random_state=42))               #set random state
])

#CATEGORICAL PIPELINE
cat_cols = X.select_dtypes(include="object").columns                            #select categorical columns
cat_transformer = Pipeline(steps=[                                              #define pipeline for categorical columns
    ('imputer', SimpleImputer(strategy='constant')),                            #fill missing values with 0
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])                       #one hot encode the categorical features

#PREPROCESSOR
preprocessor = ColumnTransformer([                                              #preprocessor for numerical columns
    ('numerical', num_transformer, num_cols),                                   #applies numerical transformer to numerical columns
    ('categorical', cat_transformer, cat_cols),                                 #applies categorical transformer to categorical columns
])

#save our preprocessor in our /models/ directory
with open(model_path + f'preprocess_pipeline.h5', 'wb') as f:
        pickle.dump(preprocessor, f)
```

# 4. Random Forest Model and Hyperparemter Tuning
Now that we have prepared our data we're ready to begin training our model.  
We will begin with a Random Forest model, and we will tune its hyperpameters using a  
Cross Validated Randomized Search as well as a Cross Validated Grid Search.  

The randomized search samples a portion of the hyperpameters in the provided grid  
to explore the hyperparemter space and give us a general idea of the best parameters  
for our model. We will use the results as a guide for our grid search which will iterate  
over each possible combination in the hyperpameter grid we supply, finding the best 
hyperparameters for the model. 

### 1. Define Base Model
First we begin by defining our base random forest model:
``` python
#define base random forest model
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', RandomForestRegressor(random_state = 42))])                   
```

### 2. Randomized Cross Validated Search 
Now we are ready to begin our randomized cross validated search to explore the best  
hyperparameters for our model. As mentioned before the Randomized Grid Search randomly  
evaluates a number of parameter combination based on the grid we define using 5 fold  
crosss validation, here we valuate 100 different hyperpameter combindations. 

``` python 
#define hyperparemeter grid for our randomized CV search
rand_search_params = {
                         'model__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400],
                         'model__max_features': ['auto', 'sqrt'],
                         'model__min_samples_leaf': [1, 2, 4, 8, 10],
                         'model__min_samples_split': [2, 5, 10, 12],
                         'model__n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
                    }
#use the randomized CV earch to explore the hyperparemeter space
rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = rand_search_params, 
                               n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
#fit random search using 5 fold cross validation
rf_random.fit(X_train, y_train)
``` 

After exploring 100 different hyperparameter combinations we can view the hyperparams that provided   
the best results: 
``` python
#print best params
rf_random.best_params_
```
*{'model__n_estimators': 1400,  
 'model__min_samples_split': 5,  
 'model__min_samples_leaf': 2,  
 'model__max_features': 'auto',  
 'model__max_depth': 50}*  

Now we will evaluate this model on our holdout set.

First we'll define an evaluation function to report RMSE and MAE: 
``` python
#define a function evaluate our model by computing the  accuracy, root mean square error, and mean absolute error
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    rmse = mean_squared_error(predictions, test_labels, squared = False)
    mae = mean_absolute_error(predictions, test_labels)
    print('Model Performance')
    print('RMSE = {:0.4f}.'.format(rmse))
    print('Mean Absolute Error = {:0.4f}.'.format(mae))
    
    return rmse
```
Now we will evaluate it on our holdout data: 
``` python
#evaluate performance of model after randomized search
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)
```
*Model Performance  
RMSE = 29259.2885.  
Mean Absolute Error = 17491.2912.*


### 3. Cross Validated Grid Search
We will further explore the results from our randomized search using its reults  
as a guide for our Cross Validated Grid Search. This will explore every possible combination   
in the hyperparameter grid we provide it.

``` python
#now we will use a GridSearch CV to further tune the best hyperparameters found in the RandomSearch
grid_search_params = { 
                     'model__n_estimators': [1000,1200,1300,1400,1500,1600,1800],
                     'model__min_samples_split': [3,5,7],
                     'model__min_samples_leaf': [2,3,5],
                     'model__max_features': ['auto'],
                     'model__max_depth': [40,45,50,55,60],
                     }


#use grid search to tune hyperparameters further
grid_search = GridSearchCV(estimator = rf_model, param_grid = grid_search_params, 
                          cv = 3, n_jobs = -1, verbose = 2)
#fit grid search
grid_search.fit(X_train, y_train)
```

Now we will view the best hyperparameters from the Grid Search: 
``` python 
#print best parameters from the grid search
grid_search.best_params_
```
*{'model__max_depth': 40,  
 'model__max_features': 'auto',  
 'model__min_samples_leaf': 2,  
 'model__min_samples_split': 5,  
 'model__n_estimators': 1600}*   

Next we will evaluate the performance of this model:
``` python
#now we will evaluate the grid search model
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)
```
*Model Performance  
RMSE = 29253.4480.  
Mean Absolute Error = 17460.5198.*  

We can see that there is a decrease in the RMSE of our model, but a slight increase in the MAE.   
This tells us that our model is making smaller errors in its predictions than before, as the RMSE  
penalizes larger errors more heavily whereas the MAE is more robust to them.

### 4. Save Random Forest with Best Hyperparemters
``` python
#finally we will save our model with the best parameters found in our Grid Search
rand_forest_path = model_path + "random_forest/"
with open(rand_forest_path + f'best_RandomForest.pickle', 'wb') as f:
        pickle.dump(best_grid, f)
``` 

Next we will move on to creating our Gradient Boosting models.




# 5. XGBoost Model
Now we are going create an XGBoost regressor and tune its hyperparameters using Optuna.  
Extreme Gradient Boosting or XGBoost is an efficient optimization algorithm for training    
gradient boosted decision trees. XGBoost combines the estimates of simpler decision trees,   
or weaker learners that are trained on the previous learners error using gradient descent   
to reduce the error of a loss function. 

Training is iterative and new trees attempt to rectify the loss of the previous 
error so that the model improves in areas where it is making mistakes. 

We will use Optuna to tune the hyperparemers of the XGBoost regressor. Optuna is a hyperparemeter  
optimization framework that automates the hyperparemeter tuning process by efficiently searching  
the hyperparameter space

### 1. Create Train and Validation Sets
We begin by creating training and validation sets and preprocessing them.  
Our Optuna objective will optimize the RMSE of our model on the holdout  
validation dataset. 

Creating and Preprocessing Training and Validation sets: 
``` python 
#split our data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8,test_size = 0.2, random_state = 42) #split training data
X_train = preprocessor.fit_transform(X_train) #preprocess X_train
X_valid = preprocessor.fit_transform(X_valid) #preprocess X_valid
```
### 2. Define Optimization Function
Now we define a function for Optuna to optimize, which will return the RMSE in this case. 
``` python
#define function to calculate RMSE 
def rmse(y_actual, y_predicted):
    return mean_squared_error(y_actual, y_predicted, squared=False)
```
### 3. Define Optuna Objective
Now we define an Optuna objective which is what our Optuna study will optimize.  
In the objective we provide Optuna with a hyperpameters grid which determines the   
hyperpameter space it will explore. 

Optuna uses a historical record of trials to determine which hyperpameters it should try next.  
Using this information it estimates the results of similar hyperparameter values in the same region,  
and continues this process, narrowing down the hyperparameter search space and effectively finding   
the best hyperparameters.

Defining our Optuna objective:
``` python
#define Optuna objective
def objective(trial):
    #define hyperparameter grid to optimize with Optuna
    params = {
                 "booster": trial.suggest_categorical('booster', ["gbtree"]),
                 "n_jobs": trial.suggest_categorical('n_jobs', [4]),
                 "n_estimators": trial.suggest_int('n_estimators', 100, 1000, 100),
                 "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.5),
                 "subsample": trial.suggest_float('subsample', 0.1, 0.5),
                 "colsample_bytree": trial.suggest_float('colsample_bytree', 0.1, 0.5),
                 "max_depth": trial.suggest_int("max_depth", 2, 20),
                 "reg_lambda": trial.suggest_float('reg_lambda', 2, 100),
                 "reg_alpha": trial.suggest_float('reg_alpha', 1, 50),
                'gamma': trial.suggest_loguniform('gamma', 1e-4,1e4),
                'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4,1e4)
            }

    #define pruning callback which will stop unpromising trials
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_0-rmse') 
    
    #define XGBRegressor base model
    model = XGBRegressor(random_state = 42,
                         **params)

    #fit model
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric=['rmse'],
              early_stopping_rounds = 10,
              verbose=2,
             callbacks =[pruning_callback])

    #return RMSE of model on validation data
    return rmse(y_valid, model.predict(X_valid))
```


### 3. Create Study and Optimize 
Now we are ready to create and run our Optuna study which will find the best   
hyperparameter combinations from the grid we provide in our objective. 
``` python
#create Optuna study
study = optuna.create_study(direction='minimize', study_name = 'XGBRegressor')
study.optimize(objective, timeout=120*60)
```

### 4. View Optuna Results
Once our Optuna Study is complete we can view the results and  hyperparameters from the best trial.
``` python
best_trial = study.best_trial                                      #select best trial from Optuna study
print('Best root mean squared error: {}'.format(best_trial.value)) #display RMSE from best trial
print('Best trial\'s parameters: ')     
for key, value in best_trial.params.items():                       #print hyperparameter values from best trial
    print('{}: {}'.format(key, value))
print('Number of finished trials:', len(study.trials))              #print number of finished trials
``` 
*Best root mean squared error: 22639.86321674742  
Best trial's parameters:   
n_estimators: 800  
learning_rate: 0.4910429061295169  
n_jobs: 4  
subsample: 0.4314552475257108  
colsample_bytree: 0.4999379628659259  
max_depth: 2  
booster: gbtree   
reg_lambda: 3.7032254243585974  
reg_alpha: 6.373633275926588  
gamma: 0.0015642079516800683  
min_child_weight: 0.05530311051359703  
Number of finished trials: 10825*  

### 5. Saving Best Model
Now we will save our model with the best hyperparameters found through our Optuna study.
``` python
#define best_model with optimized hyperparameters
best_params = study.best_trial.params       #select best hyperparameters
best_model = XGBRegressor(random_state=42,  #create XGBRegressor with best hyperparams
                         **best_params)

#finally we save our best XGBoost model
XGBoost_path = model_path + "XGBoost/"
with open(XGBoost_path + f'best_XGBoost.pickle', 'wb') as f:
        pickle.dump(best_model, f)
```



# 6. CatBoost Model
Now we will create a CatBoost regressor, CatBoost is another Gradient Boosting model similar to XGBoost,
but is has some differenes in the ways it grows leaves, splits at nodes, handles missing and categorical values, 
as well as other features. 

We will also tune this model using Optuna in the same manner as our XGBoost model. For a brief explanation regarding Optuna[refer to the 
XGBoost section.](https://github.com/wct432/kaggle_notebooks/blob/main/housing_prices/README.md#3-define-optuna-study)

### 1. Create CatBoost Model and Optuna Objective
Similar to before we first prepare train and validation sets, define a RMSE function for Optuna to optimize, and preprocess our data using
the Sklearn preprocess pipeline we created in our Random Forest notebook. 

After preparing our data we're ready to define our Optuna objective: 
``` python
#define Optuna objective
def objective(trial):
    #hyperparameters to optimize with optuna
    params = {
                    'depth': trial.suggest_int('depth',6,10,1)    ,                 
                    'iterations': trial.suggest_int('iterations',100,1600,100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                    'l2_leaf_reg': trial.suggest_int('l2_leaf_reg',0,20,5),
                    'random_strength': trial.suggest_float('random_strength',0,2.5)
                }
        
    #create CatBoost regressor model
    model = CatBoostRegressor(random_seed = 42,
                              early_stopping_rounds=20,         
                              grow_policy='Depthwise',
                              leaf_estimation_method='Newton',
                              bootstrap_type='Bernoulli',
                              thread_count=-1,
                              verbose=2,
                              loss_function='RMSE',
                              eval_metric='RMSE',
                              od_type='Iter',
                              **params)
                              
    #fit model
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              early_stopping_rounds = 10,
              verbose=2)

    #return RMSE of model on validation data
    return rmse(y_valid, model.predict(X_valid))
```
### 2. Create Optuna Study and Optimize
Now we are ready to create and run our Optuna study which will find the best   
hyperparameter combinations from the grid we provide in our objective. 

``` python
#create Optuna study
study = optuna.create_study(direction='minimize', study_name = 'CatBoostRegressor')
study.optimize(objective, timeout=120*60)
```

### 3. View Optuna Results
Once our Optuna Study is complete we can view the results and  hyperparameters from the best trial.

``` python
best_trial = study.best_trial
print('Best root mean squared error: {}'.format(best_trial.value))
print('Best trial\'s parameters: ')
for key, value in best_trial.params.items():
    print('{}: {}'.format(key, value))
print('Number of finished trials:', len(study.trials))
```
*Best root mean squared error: 20933.61446938353  
Best trial's parameters:   
depth: 6  
iterations: 1300  
learning_rate: 0.3797571932288995  
l2_leaf_reg: 5  
random_strength: 0.6568521493639833  
Number of finished trials: 4912*  

### 4. Saving Best Model
Now we will save our model with the best hyperparameters found through our Optuna study.
``` python 
#define best_model with optimized hyperparameters   
best_params = study.best_trial.params           #select best hyperparameters
best_model = CatBoostRegressor(random_state=42, #create CatBoostRegressor with best hyperparams
                         **best_params)


#save our CatBoost model to our /models/ subdirectory
CatBoost_path = model_path + "CatBoost/"
with open(CatBoost_path + f'best_CatBoost.pickle', 'wb') as f:
        pickle.dump(best_model, f)
```



# 7. Create and Fit Ensemble Model
Now we will create an ensemble model using Scikit-Learn's Stacking Regressor.   

The Stacking Regressor is an ensemble technique that create a meta-regressor, or meta-learner 
by stacking the base models predictions as input into the meta-regressor to create an ensemble model. 

We will use Ridge CV as our meta-regressor, which is a cross validated regression model that   
is can estimate the coefficients of variables in situations where the independent variables are correlated.

We will load our Random Forest, XGBoost, and CatBoost regression models from our model directory, which will   
be used as the base models for our Stacking Regressor. We will also add a simple linear regression model as another  
weak-learner, as I found performance benefits from doing so. 


### 1. Create Ensemble Model 
After loading our models from our directory we are ready to create our Stacking Regressor. 

``` python
#define estimators for ensemble model
estimators = [
        ('rf_model', rf_regressor),
        ('xgb_model', xgb_regressor),
        ('cat_model', catboost_regressor),
        ('linear_model',LinearRegression())]

#define stacking regressor to create an ensemble model
stack = StackingRegressor(estimators=estimators,
                            final_estimator=RidgeCV(), cv=5,
                            passthrough = False,verbose = 2,
                            n_jobs=-1)
```

### 2. Fit Ensemble Model
We can now fit our ensemble model by calling its fit method.
``` python
#fit stacking regressor
stack.fit(X_train,y_train)
```

### 3. Create Final Predictions and Save Ensemble Model
Once our ensemble model is finished training we can make our final predictions
on the testing dataset for the our Kaggle competition!

``` python
#make final predictions to submit to competition
final_pred = stack.predict(testing_set)
#output predictions as df for Kaggle competition
final_pred = pd.DataFrame({'Id': test_indices,
                       'SalePrice': final_pred})
final_pred.to_csv(data_path + 'predictions/ensemble_submission.csv', index=False)

#finally we save our ensemble model
with open(ensemble_path + f'ensemble_model.pickle', 'wb') as f:
        pickle.dump(stack, f)
```
After submitting my prediction to the competition I was pleased to find the model had achieved  
a RMSE of 13,969 on the competition data and a score in the top 1%!
