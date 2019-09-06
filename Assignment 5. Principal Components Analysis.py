
# coding: utf-8

# In[68]:


# Random Seed Value
RANDOM_SEED = 11
RANDOM_SEED_MODEL = 111

# No. of Folds for Cross-val
N_FOLDS = 10

# Import packages
import numpy as np
import pandas as pd
import os
import time    
from shutil import copyfileobj
from six.moves import urllib       
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  

# Sklearn packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.metrics import f1_score
from sklearn.datasets.base import get_data_home  

# Datapackage from Sklearn
from sklearn.datasets import fetch_mldata


# In[69]:


# Per book; definition for function for displaying observations
def plot_digits(instances, images_per_row = 10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis('off')


# In[70]:


# Import data pt. 1
def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)


# In[71]:


# Import data pt. 2
fetch_mnist()
mnist = fetch_mldata("MNIST original")
mnist


# In[72]:


# define arrays 
mnist_X, mnist_y = mnist['data'], mnist['target']


# In[73]:


# Create freq dist for 60k observations for training model building
mnist_y_0_59999_df = pd.DataFrame({'label': mnist_y[0:59999,]}) 
print('\nFrequency distribution for model building')
print(mnist_y_0_59999_df['label'].value_counts(ascending = True)) 

# Create freq dist for remaining 10k observations for testing
mnist_y_60000_69999_df = pd.DataFrame({'label': mnist_y[60000:69999,]}) 
print('\nFrequency distribution holdout sample')
print(mnist_y_60000_69999_df['label'].value_counts(ascending = True)) 


# In[75]:


# print out copy of the digits
with PdfPages('Handwritten digits For Model.pdf') as pdf:
    for idigit in range(0,10):
        idigit_indices =             mnist_y_0_59999_df.index[mnist_y_0_59999_df.label == idigit]   
        show_indices = resample(idigit_indices, n_samples=100, 
                                replace = False, 
                                random_state = RANDOM_SEED).sort_values()       
        plt.figure(0)
        plt.suptitle('Example for MNIST Digit ' + str(idigit))
        for j in range(0,10):
            row_begin_index = j * 10
            row_end_index = row_begin_index + 10
            this_row_indices = show_indices[row_begin_index:row_end_index]
            example_images = np.r_[mnist_X[this_row_indices]]
            plt.subplot2grid((10,1), (j,0), colspan=1)
            plot_digits(example_images, images_per_row=10)
            row_begin_index = row_end_index + 1
        pdf.savefig()  
        plt.close()   

with PdfPages('Handwritten digits For Test.pdf') as pdf:
    for idigit in range(0,10):
        idigit_indices = 60000 +         mnist_y_60000_69999_df.index[mnist_y_60000_69999_df.label == idigit]
        show_indices = resample(idigit_indices, n_samples=100, 
                                replace = False, 
                                random_state = RANDOM_SEED).sort_values()       
        plt.figure(0)
        plt.suptitle('Example for MNIST Digit ' + str(idigit))
        for j in range(0,10):
            row_begin_index = j * 10
            row_end_index = row_begin_index + 10
            this_row_indices = show_indices[row_begin_index:row_end_index]
            example_images = np.r_[mnist_X[this_row_indices]]
            plt.subplot2grid((10,1), (j,0), colspan=1)
            plot_digits(example_images, images_per_row=10)
            row_begin_index = row_end_index + 1
        pdf.savefig()  
        plt.close()   
        
with PdfPages('Plot of Digits.pdf') as pdf:
    fig = plt.figure(figsize=(9,9))
    example_images = np.r_[mnist_X[:12000:600], mnist_X[13000:30600:600], mnist_X[30600:60000:590]]
    plot_digits(example_images, images_per_row=10)
    #save_fig("more_digits_plot")
    pdf.savefig(fig)


# In[76]:


# Training data and shape of array
model_y = np.r_[mnist_y_0_59999_df]
model_X = np.r_[mnist_X[0:59999,]]
model_data = np.concatenate((model_y.reshape(-1, 1), model_X), axis = 1)

print('\nShape of model_data:', model_data.shape) 

# Test data and shape of array
holdout_y = np.r_[mnist_y_60000_69999_df]
holdout_X = np.r_[mnist_X[60000:69999,]]
holdout_data = np.concatenate((holdout_y.reshape(-1, 1), 
                               holdout_X), axis = 1)

print('\nShape of holdout_data:', holdout_data.shape)


# In[77]:


# Shuffle Rows in MNIST data
np.random.seed(RANDOM_SEED)
np.random.shuffle(model_data)

np.random.seed(RANDOM_SEED)
np.random.shuffle(holdout_data)


# In[78]:


# Model and Array Setup
names = ["Random Forest"]
classifiers = [RandomForestClassifier(n_estimators=10, max_features='sqrt', 
                            bootstrap=True)]

cv_results = np.zeros((N_FOLDS, len(names)))


# In[79]:


# This is the start of the random forest classifier test using cross-val
start_time = time.clock()

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
index_for_fold = 0   
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
    
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

    index_for_method = 0  
    for name, clf in zip(names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        clf.fit(X_train, y_train)  
        y_test_predict = clf.predict(X_test)
        fold_method_result = f1_score(y_test,y_test_predict,average='weighted') 
        print('F1 Score:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 F1 Score', sep = '')     
print(cv_results_df.mean())   

end_time = time.clock()
runtime = end_time - start_time    
 
print('\nRuntime for Random Forest:', runtime)  


# In[80]:


# PCA with timer per assignment conditions                 
start_time_pca = time.clock()  

pca = PCA(n_components=0.95)
reduced = pca.fit_transform(mnist_X)
                     
end_time_pca = time.clock()

runtime_pca = end_time_pca - start_time_pca    
runtime_pca    


# In[81]:


# Reduced data train and test model
train = np.concatenate((model_y.reshape(-1, 1), reduced[0:59999,]), axis = 1)
test = np.concatenate((holdout_y.reshape(-1, 1), 
                               reduced[60000:69999,]), axis = 1)
                              
# As with the last segment, shuffle Rows in MNIST data
np.random.seed(RANDOM_SEED)
np.random.shuffle(train)

np.random.seed(RANDOM_SEED)
np.random.shuffle(test)

print('\nShape of train:', train.shape)    
print('\nShape of test:', test.shape) 

# random forest classifier test using cross-val second time
cv_resultsPCA = np.zeros((N_FOLDS, len(names)))

start_time_reducedRF = time.clock()

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
index_for_fold = 0  
for train_index, test_index in kf.split(train):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
    
    X_train = train[train_index, 1:train.shape[1]]
    X_test = train[test_index, 1:train.shape[1]]
    y_train = train[train_index, 0]
    y_test = train[test_index, 0]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, clf in zip(names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        clf.fit(X_train, y_train)  
        y_test_predict = clf.predict(X_test)
        fold_method_result = f1_score(y_test,y_test_predict,average='weighted') 
        print('F1 Score:', fold_method_result)
        cv_resultsPCA[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1

cv_resultsPCA_df = pd.DataFrame(cv_resultsPCA)
cv_resultsPCA_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 F1 Score', sep = '')     
print(cv_resultsPCA_df.mean())   

end_time_reducedRF = time.clock()

runtime_reducedRF = end_time_reducedRF - start_time_reducedRF  
runtime_reducedRF


# In[82]:


# TOTAL PCA runtime analysis
runtime_total_pca = runtime_pca + runtime_reducedRF
runtime_total_pca


print('\nRuntime for Random Forest:', runtime)   
print('\nRuntime for Dimension Reduced Random Forest:', runtime_total_pca)


# In[83]:


# Pure Random Forest Classifier
rnd_clf = RandomForestClassifier(n_estimators=10, max_features='sqrt', 
                             bootstrap=True)
rnd_clf.fit(model_data[:,1:model_data.shape[1]], model_data[:,0])
y_predrf = rnd_clf.predict(holdout_X)

rrnd_clf = RandomForestClassifier(n_estimators=10, max_features='sqrt', 
                            bootstrap=True)
rrnd_clf.fit(train[:,1:train.shape[1]], train[:,0])
y_predrrf = rrnd_clf.predict(test[:,1:test.shape[1]])


pure_rf = f1_score(y_predrf, holdout_y, average='weighted')

print('\nPure Random Forest Runtime:', pure_rf)                                                                     

