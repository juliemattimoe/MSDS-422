# Jump-Start for the Bank Marketing Study
# as described in Marketing Data Science: Modeling Techniques
# for Predictive Analytics with R and Python (Miller 2015)

# jump-start code revised by Thomas W. Milller (2017/09/26)

# Scikit Learn documentation for this assignment:
# http://scikit-learn.org/stable/auto_examples/classification/
#   plot_classifier_comparison.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB.score
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.LogisticRegression.html
# http://scikit-learn.org/stable/modules/model_evaluation.html 
# http://scikit-learn.org/stable/modules/generated/
#  sklearn.model_selection.KFold.html

# prepare for Python version 3x features and functions
# comment out for Python 3.x execution
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score   
from sklearn.model_selection import KFold

# load csv file into a DataFrame
def load_csv(filename):
    data = pd.read_csv(filename, sep = ';')
    return data

# prints shape of data for later in the array
# then it ommit the associate name
def print_shape(data, name = None):
    if name != None:
        data.name = name
        print("The shape of data, {}, is: {}".format(name, str(data.shape)))
    else:
        print("The shape of data is: {}".format(str(data.shape)))
    
# initial work with the smaller data set
bank = pd.read_csv('bank.csv')  # start with smaller data set
# examine the shape of original input data
print(bank.shape)

# look at the list of column names, note that y is the response
list(bank.columns.values)

# look at the beginning of the DataFrame
bank.head()

# mapping function to convert text no/yes to integer 0/1
def map_to_binary(dataframe, feature):
    mapped_df = pd.DataFrame()
    impute_to_binary = {'no' : 0, 'yes' : 1}
    mapped_df = dataframe[feature].map(impute_to_binary)
    return mapped_df
    
# define binary variable for having credit in default
default = map_to_binary(bank, 'default')

# define binary variable for having a mortgage or housing loan
housing = map_to_binary(bank, 'housing')

# define binary variable for having a personal loan
loan = map_to_binary(bank, 'loan')

# define response variable to use in the model
response = map_to_binary(bank, 'response')

# gather three explanatory variables and response into a numpy array 
# here we use .T to obtain the transpose for the structure we want
model_data = np.array([np.array(default), np.array(housing), np.array(loan), 
    np.array(response)]).T

# examine the shape of model_data, which we will use in subsequent modeling
print_shape(model_data)

# seed generator
np.random.seed(RANDOM_SEED)
np.random.shuffle(model_data)

#  examine the shape of original input data
print_shape(model_data)

# move into classifier models with logistic regression and naive bayes
classifier_names = ["Logistic_Regression", "Naive_Bayes"]
classifiers = [LogisticRegression(), BernoulliNB(alpha=1.0, binarize=0.5, 
                           class_prior = [0.5, 0.5], fit_prior=False)]

# assess how the results will generalize to an independent data set using a 10-fold cross validation
N_FOLDS = 15

# create array for results list of 15-fold cross validation
crossvalidation_results = np.zeros((N_FOLDS, len(classifier_names)))

# kf set-up with random seed generator
kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)

# fold count initialized to 0 to emit value continuously
index_for_fold = 0 

# split data to fit model
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
    # 0:model_data.shape[1]-1 slices for explanatory variables,
    X_train = model_data[train_index, 0:model_data.shape[1]-1]
    X_test = model_data[test_index, 0:model_data.shape[1]-1]
    
    # model_data.shape[1]-1 is the index for the response variable
    y_train = model_data[train_index, model_data.shape[1]-1]
    y_test = model_data[test_index, model_data.shape[1]-1]
    
    # prints structure of data after split for x, y 
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)
    
    # old count initialized to 0 and performs predictions
    index_for_method = 0
    for name, clf in zip(classifier_names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        
    # use train data set
        clf.fit(X_train, y_train) 
        
    # performs predictions
        y_test_predict = clf.predict_proba(X_test)
        
    # calculates ROC AUC score, stores results in cv_results
        fold_method_result = roc_auc_score(y_test, y_test_predict[:,1]) 
        print('Area under ROC curve:', fold_method_result)
        crossvalidation_results[index_for_fold, index_for_method] = fold_method_result
        
        
   # moved old count from 0 to 1 so each loop will be the next classifier
        index_for_method += 1
        
    # moved old count from 0 to 1 so each loop will be the next split
    index_for_fold += 1

# results from 15-fold cross-validation from 0 to 14 
crossvalidation_results_df = pd.DataFrame(crossvalidation_results)
crossvalidation_results_df.columns = classifier_names
with open("cv-results-df.txt", "w") as text_file:
    text_file.write('\nResults from '+ str(N_FOLDS) + '-fold cross-validation\n'+
                     '\nMethod Area under ROC Curve:\n'+ 
                     str(crossvalidation_results_df))

# mean of ROC AUC evaluation results 
print('\n----------------------------------------------')
print('\nAverage results from {}-fold cross-validation\n\nMethod Area under ROC Curve:\n{}'
      .format(str(N_FOLDS),str(crossvalidation_results_df.mean())), sep = '')     
print('\nMean of cross validation result: {}'.format(crossvalidation_results_df.mean())) 
with open("cv-results-df-mean.txt", "w") as text_file:
    text_file.write('\nAverage results from {}-fold cross-validation\n\nMethod Area under ROC Curve:\n{}'
                    .format(str(N_FOLDS),str(crossvalidation_results_df.mean())))