# Boston Housing Study (Python)
# using data from the Boston Housing Study case
# as described in "Marketing Data Science: Modeling Techniques
# for Predictive Analytics with R and Python" (Miller 2015)

# Here we use data from the Boston Housing Study to evaluate
# regression modeling methods within a cross-validation design.

# program revised by Thomas W. Milller (2017/09/29)

# Scikit Learn documentation for this assignment:
# http://scikit-learn.org/stable/modules/model_evaluation.html 
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.model_selection.KFold.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.LinearRegression.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.Ridge.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.Lasso.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.ElasticNet.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.metrics.r2_score.html

# Textbook reference materials:
# Geron, A. 2017. Hands-On Machine Learning with Scikit-Learn
# and TensorFlow. Sebastopal, Calif.: O'Reilly. Chapter 3 Training Models
# has sections covering linear regression, polynomial regression,
# and regularized linear models. Sample code from the book is 
# available on GitHub at https://github.com/ageron/handson-ml

# prepare for Python version 3x features and functions
# comment out for Python 3.x execution
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation
from sklearn.model_selection import KFold

# Set up Corr_plot
def corr_plot(df_corr):
    corr = df_corr.corr()
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    sns.heatmap(boston_input.corr(), mask=top, cmap='coolwarm', annot=True, fmt=".2f")
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 0)
    plt.show()

# read data for the Boston Housing Study
# creating data frame restdata
boston_input = pd.read_csv('boston.csv')

# check the pandas DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())

print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())

# print stats and export it to a saved txt file 
def print_data_info_save_to_file(data, dataname):
    print('\n---------{} data informations----------\n'.format(dataname))
    print('\n{} data shape: {}'.format(dataname, data.shape))
    print('\n{} data dtypes: {}'.format(dataname, data.dtypes))
    print('\n{} data column values: {}'.format(dataname, data.columns.values)) 
    print('\n{} data first few rows: {}'.format(dataname, data.head())) 
    print('\n{} data look at end of data: {}'.format(dataname, data.tail()))
    print('\n{} data descriptive statistics: {}'.format(dataname, data.describe()))
    print('\n{} data numerical correlations: {}'.format(dataname, data.corr(method='pearson')))
    with open("{}_data_descriptive_information.txt".format(dataname), "w") as text_file:
        text_file.write('\n---------{} data informations----------\n'.format(dataname)+
                        '\n{} data shape: {}'.format(dataname, str(data.shape)) +
                        '\n{} data dtypes: {}'.format(dataname, str(data.dtypes)) +
                        '\n{} data column values: {}'.format(dataname, str(data.columns.values)) + 
                        '\n{} data first few rows: {}'.format(dataname, str(data.head()))+ 
                        '\n{} data look at end of data: {}'.format(dataname, str(data.tail()))+
                        '\n{} data descriptive statistics: {}'.format(dataname, str(data.describe()))+ 
                        '\n{} data information: {}'.format(dataname, str(data.info()))+
                        '\n{} data numerical correlation: {}'.format(dataname, str(boston_input.corr(method='pearson'))))
print_data_info_save_to_file(boston_input, 'boston_input')


# plot the numerical correlations
boston_input_features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rooms', 'age', 'dis', 'rad', 'tax', 'ptratio', 'lstat', 'mv']
corr_plot(boston_input)   

# histogram and density plots for review with saved image
def hist_density_plots(data, dataname, title, features):
    d = data
    f = features
    g = pd.melt(d, value_vars = f)
    gm = sns.FacetGrid(g, col='variable', col_wrap = 4, sharex = False, sharey = False)
    gm = gm.map(sns.distplot, 'value')
    plt.title(title)
    plt.title(title)
    plt.savefig('Hist-Density-{}.pdf'.format(dataname), 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None)
    plt.show()
    plt.close()
hist_density_plots(boston_input, 'Boston-Input','Boston Input Histogram/Density Plots',
                   boston_input_features)
   
# scatter matrix for review with saved image
def scatter_matrix(data, dataname, title, features):
    d = data
    sns.pairplot(d)
    plt.title(title)

    plt.savefig('Scatte-Matrix-{}.pdf'.format(dataname), 
                bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                orientation='portrait', papertype=None, format=None, 
                transparent=True, pad_inches=0.25, frameon=None)
    plt.show()
    #plt.close()
scatter_matrix(boston_input, 'Boston-Input', 'Boston Input Scatter Matrix',
               boston_input_features)    
    

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())

# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,\
    boston.crim,\
    boston.zn,\
    boston.indus,\
    boston.chas,\
    boston.nox,\
    boston.rooms,\
    boston.age,\
    boston.dis,\
    boston.rad,\
    boston.tax,\
    boston.ptratio,\
    boston.lstat]).T

# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)

# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))
# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)

# Standardize the model from preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)

# At this time, we choose three regression models 
names = ['Lasso_Regression','Linear_Regression', 
         'Ridge_Regression'] 

# list of regressors
regressors = [Lasso(alpha = 0.1, max_iter=10000, tol=0.01, 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     random_state = RANDOM_SEED),
              LinearRegression(fit_intercept = SET_FIT_INTERCEPT),
              Ridge(alpha = 1, solver = 'cholesky', 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     normalize = False, 
                     random_state = RANDOM_SEED)]

# Next, we use cross-validation
# fifteen-fold cross-validation used here as was the case with Assignment 2 then print it to a file
N_FOLDS = 15

# cv_results for the cross-validation results
cv_results = np.zeros((N_FOLDS, len(names)))

# kf to set up KFold 
kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)

# Look at fold/observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold)
    X_train = model_data[train_index, 1:model_data.shape[1]]
    X_test = model_data[test_index, 1:model_data.shape[1]]
    y_train = model_data[train_index, 0]
    y_test = model_data[test_index, 0]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    # Initialize to zero
    index_for_method = 0 
    for name, reg_model in zip(names, regressors):
        print('\nRegression model evaluation for:', name)
        print('  Scikit Learn method:', reg_model)
        reg_model.fit(X_train, y_train)  # fit on the train set for this fold
        print('Fitted regression intercept:', reg_model.intercept_)
        print('Fitted regression coefficients:', reg_model.coef_)
 
        y_test_predict = reg_model.predict(X_test)
        print('Coefficient of determination (R-squared):',
              r2_score(y_test, y_test_predict))
        fold_method_result = sqrt(mean_squared_error(y_test, y_test_predict))
        print(reg_model.get_params(deep=True))
        print('Root mean-squared error:', fold_method_result,
              '\n--------------------------------------------------------\n')
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
    
    index_for_fold += 1
    
# Store the cross-validation results
cv_results_df = pd.DataFrame(cv_results)

# cv_results_df and column names from DataFrame 
cv_results_df.columns = names
with open("cv-results.txt", "w") as text_file:
    text_file.write('\nCross validation results:\n'+
                    str(cv_results_df)+
                    '\nCross validation results column names:\n'+
                    str(names))

# Export the results of the cross-validation as cv_results and regression results as cv_results_df_mean
pd.set_option('precision', 5)
print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      'in standardized units (mean 0, standard deviation 1)\n',
      '\nMethod Root mean-squared error', sep = '') 
print(cv_results_df.mean())   
with open("cv-results-df-mean.txt", "w") as text_file:
    text_file.write('\nAverage results from '+ str(N_FOLDS) + '-fold cross-validation\n'+
                    'in standardized units (mean 0, standard deviation 1)\n'+
                     '\nMethod Root mean-squared error:\n'+ 
                     str(cv_results_df.mean()))