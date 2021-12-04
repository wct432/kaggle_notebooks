
# Housing Price Competitino 

## Introduction
The goal of this project is housing price prediction in Kaggle's Housing Prices Comeptition. 

The goal of the competition is predicting the price of residential homes in Ames, Iowa 
and is scored using the RMSE. The dataset contains 79 features. 

I approached the problem using an ensemble model combining random forest, XGBoost, CatBoost and linear regression models.

The hyperparameters of the decision tree were tuned using a cross-validated random search and grid search,
and the XGBoost and CatBoost models were tuned using Optuna. 


## Exploratory Analysis
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
# select columns that contain null values and the sum of the null's as a tuple
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

## Plots and Visualizations

### 1. Mean Housing Price per Neighborhood 
<details>
<summary>Click to display code!</summary>

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




2. Feature/Target Correlation Heatmap
Here we will create a heatmap that shows the amount of correlation each feature has with 
Housing Price, our target variable. 

![feature/target correlation](/../images/images/feature_target_correlation.png?raw=true)

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
```
