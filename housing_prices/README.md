
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
each column, and other information about the data like the number of n/a values. 

Then we create some visualizations to help us learn about the relationship between the variables.

### Visualizations

1. Feature/Target Correlation

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
