
# coding: utf-8

# In[1]:


# import necessary packages
import numpy as np
import time
from tabulate import tabulate

# import tensorflow to use to build neural network
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


# define the number of epochs that we want to run
N_INPUTS = 28*28  
N_OUTPUTS = 10    


# In[3]:


# Reset graph
def reset_graph(seed=111):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)  

def two_layer_NN(n_hidden1, n_hidden2, activate):
    reset_graph()
    
    tf.set_random_seed(111)
    
    # placeholder nodes
    X = tf.placeholder(tf.float32, shape=(None, N_INPUTS), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y") 
    
    
    # Create the network with cost function used to train the network
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                                activation=activate)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                                activation=activate)
        logits = tf.layers.dense(hidden2, N_OUTPUTS, name="outputs") 
    
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, 
                                                                logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    
    # define a GradientDescentOptimizer per chapter 10
    learning_rate = 0.01
    
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
        
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  
    
    # create a node to initialize all variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(save_relative_paths=True)  
    
    # define the number of epochs that we want to run 
    n_epochs = 25
    batch_size = 50
    
    # Per the assignment, need to get time recorded 
    start_time = time.clock()
    
    # Random sampling 
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.validation.images, 
                                                y: mnist.validation.labels})
            print(epoch, "Train accuracy:", acc_train, 
                          "Test accuracy:", acc_test)
    
        save_path = saver.save(sess, './model_final.ckpt')
    
    # final score      
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        accuracy = accuracy.eval(feed_dict={X: mnist.test.images, 
                                        y: mnist.test.labels})
    # time measurement
    stop_time = time.clock()
    runtime = stop_time - start_time  
    
    return accuracy, runtime, acc_test


# In[4]:


def five_layer_NN(n_hidden1, n_hidden2, n_hidden3, n_hidden4,
                    n_hidden5, activate):
    # Reset Graph
    reset_graph()
    
    tf.set_random_seed(111)
    
    # placeholder nodes round 2
    X = tf.placeholder(tf.float32, shape=(None, N_INPUTS), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y") 
    
    
    # Neural Network
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                                activation=activate)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                                activation=activate)
        hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3",
                                activation=activate)
        hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4",
                                activation=activate)
        hidden5 = tf.layers.dense(hidden4, n_hidden5, name="hidden5",
                                activation=activate)
        logits = tf.layers.dense(hidden5, N_OUTPUTS, name="outputs") 
    
    # Train Neural Network
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, 
                                                                logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    
    # define a GradientDescentOptimizer per chapter 10
    learning_rate = 0.01
    
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
        
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  
    
    # create a node to initialize all variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(save_relative_paths=True) 

    # define the number of epochs that we want to run 
    n_epochs = 25
    batch_size = 50
    
    # Per the assignment, need to get time recorded 
    start_time = time.clock()
    
    # Random sampling 
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.validation.images, 
                                                y: mnist.validation.labels})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    
        save_path = saver.save(sess, './model_final.ckpt')
        
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        accuracy = accuracy.eval(feed_dict={X: mnist.test.images, 
                                        y: mnist.test.labels})
    
    # time measurement
    stop_time = time.clock()
    runtime = stop_time - start_time  
    
    return accuracy, runtime, acc_test


# In[5]:


# Read in dataset
mnist = input_data.read_data_sets("/tmp/data/")

# print shape of data
X_train = mnist.train.images
print("Shape of Training data: ", X_train.shape)
X_validate = mnist.validation.images
print("Shape of Validate data: ", X_validate.shape)
X_test = mnist.test.images
print("Shape of Test data: ", X_test.shape)


# In[6]:


# Neural Network Model executuion

# Basic NN structure
# Model 1: 2 layer; hidden (300,100); activation: reLU 
n_hidden1_M1 = 300
n_hidden2_M1 = 100
activate_M1 = tf.nn.relu
 
accuracy_M1, runtime_M1, acc_trainM1 = two_layer_NN(n_hidden1_M1, n_hidden2_M1, 
                                        activate_M1)

# Basic NN structure
# Model 2: 2 layer; hidden (300,100); activation: Elu 
n_hidden1_M2 = 300
n_hidden2_M2 = 100
activate_M2 = tf.nn.elu

accuracy_M2, runtime_M2, acc_trainM2 = two_layer_NN(n_hidden1_M2, n_hidden2_M2, 
                                        activate_M2)

# Basic NN structure
# Model 3: 2 layer; hidden (300,100); activation: tanh 
n_hidden1_M3 = 300
n_hidden2_M3 = 100
activate_M3 = tf.nn.tanh

accuracy_M3, runtime_M3, acc_trainM3 = two_layer_NN(n_hidden1_M3, n_hidden2_M3,
                                        activate_M3)

# Basic NN structure
# Model 4: 2 layer; hidden (200,200); activation: reLU 
n_hidden1_M4 = 200
n_hidden2_M4 = 200
activate_M4 = tf.nn.relu

accuracy_M4, runtime_M4, acc_trainM4 = two_layer_NN(n_hidden1_M4, n_hidden2_M4, 
                                        activate_M4)

# Basic NN structure
# Model 5: 2 layer; hidden (200,200); activation: Elu 
n_hidden1_M5 = 200
n_hidden2_M5 = 200
activate_M5 = tf.nn.elu

accuracy_M5, runtime_M5, acc_trainM5 = two_layer_NN(n_hidden1_M5, n_hidden2_M5, 
                                        activate_M5)

# Basic NN structure
# Model 6: 2 layer; hidden (200,200); activation: tanh 
n_hidden1_M6 = 200
n_hidden2_M6 = 200
activate_M6 = tf.nn.tanh

accuracy_M6, runtime_M6, acc_trainM6 = two_layer_NN(n_hidden1_M6, n_hidden2_M6, 
                                        activate_M6)

# Basic NN structure
# Model 7: 5 layer; hidden (100,100,100,100,100); activation: relu 
n_hidden1_M7 = 100
n_hidden2_M7 = 100
n_hidden3_M7 = 100
n_hidden4_M7 = 100
n_hidden5_M7 = 100
activate_M7 = tf.nn.relu

accuracy_M7, runtime_M7, acc_trainM7 = five_layer_NN(n_hidden1_M7, n_hidden2_M7, 
                        n_hidden3_M7, n_hidden4_M7, n_hidden5_M7, activate_M7)
                        
                        
# Basic NN structure
# Model 8: 5 layer; hidden (100,100,100,100,100); activation: Elu 
n_hidden1_M8 = 100
n_hidden2_M8 = 100
n_hidden3_M8 = 100
n_hidden4_M8 = 100
n_hidden5_M8 = 100
activate_M8 = tf.nn.elu

accuracy_M8, runtime_M8, acc_trainM8 = five_layer_NN(n_hidden1_M8, n_hidden2_M8, 
                        n_hidden3_M8, n_hidden4_M8, n_hidden5_M8, activate_M8)

# Basic NN structure
# Model 9: 5 layer; hidden (100,100,100,100,100); activation: tanh 
n_hidden1_M9 = 100
n_hidden2_M9 = 100
n_hidden3_M9 = 100
n_hidden4_M9 = 100
n_hidden5_M9 = 100
activate_M9 = tf.nn. tanh

accuracy_M9, runtime_M9, acc_trainM9 = five_layer_NN(n_hidden1_M9, n_hidden2_M9, 
                        n_hidden3_M9, n_hidden4_M9, n_hidden5_M9, activate_M9)


# Basic NN structure
# Model 10: 5 layer; hidden (200,200,200,200,200); activation: relu 
n_hidden1_M10 = 200
n_hidden2_M10 = 200
n_hidden3_M10 = 200
n_hidden4_M10 = 200
n_hidden5_M10 = 200
activate_M10 = tf.nn.relu

accuracy_M10, runtime_M10, acc_trainM10 = five_layer_NN(n_hidden1_M10, 
                            n_hidden2_M10, n_hidden3_M10, n_hidden4_M10, 
                            n_hidden5_M10, activate_M10)

# Basic NN structure
# Model 11: 5 layer; hidden (200,200,200,200,200); activation: Elu 
n_hidden1_M11 = 200
n_hidden2_M11 = 200
n_hidden3_M11 = 200
n_hidden4_M11 = 200
n_hidden5_M11 = 200
activate_M11 = tf.nn. elu

accuracy_M11, runtime_M11, acc_trainM11 = five_layer_NN(n_hidden1_M11, 
                            n_hidden2_M11, n_hidden3_M11, n_hidden4_M11, 
                            n_hidden5_M11, activate_M11)            

# Basic NN structure
# Model 12: 5 layer; hidden (200,200,200,200,200); activation: tanh 
n_hidden1_M12 = 200
n_hidden2_M12 = 200
n_hidden3_M12 = 200
n_hidden4_M12 = 200
n_hidden5_M12 = 200
activate_M12 = tf.nn.tanh

accuracy_M12, runtime_M12, acc_trainM12 = five_layer_NN(n_hidden1_M12, 
                            n_hidden2_M12, n_hidden3_M12, n_hidden4_M12, 
                            n_hidden5_M12, activate_M12)


# In[7]:


print("Accuracy Score for Model 1: ", accuracy_M1)
print("Run Time for Model 1: ", runtime_M1)  

print("Accuracy Score for Model 2: ", accuracy_M2)
print("Run Time for Model 2: ", runtime_M2) 

print("Accuracy Score for Model 3: ", accuracy_M3)
print("Run Time for Model 3: ", runtime_M3)  

print("Accuracy Score for Model 4: ", accuracy_M4)
print("Run Time for Model 4: ", runtime_M4)  

print("Accuracy Score for Model 5: ", accuracy_M5)
print("Run Time for Model 5: ", runtime_M5)  

print("Accuracy Score for Model 6: ", accuracy_M6)
print("Run Time for Model 6: ", runtime_M6)  

print("Accuracy Score for Model 7: ", accuracy_M7)
print("Run Time for Model 7: ", runtime_M7)  

print("Accuracy Score for Model 8: ", accuracy_M8)
print("Run Time for Model 8: ", runtime_M8)

print("Accuracy Score for Model 9: ", accuracy_M9)
print("Run Time for Model 9: ", runtime_M9) 

print("Accuracy Score for Model 10: ", accuracy_M10)
print("Run Time for Model 10: ", runtime_M10) 

print("Accuracy Score for Model 11: ", accuracy_M11)
print("Run Time for Model 11: ", runtime_M11) 

print("Accuracy Score for Model 12: ", accuracy_M12)
print("Run Time for Model 12: ", runtime_M12) 


# In[11]:


# Per the assignment page, create an output table

col_labels = ['Number of Layers', 'Nodes per Layer', 
                                'Activation Function', 'Processing Time',
                                'Training Set Accuracy', 'Test Set Accuracy']
                                
table_vals = [[2, "(" + str(n_hidden1_M1) + "," + str(n_hidden2_M1) + ")", 
                str(activate_M1).split(" ")[1], round(runtime_M1,2), 
                round(acc_trainM1,3), round(accuracy_M1, 3)],
               [2, "(" + str(n_hidden1_M2) + "," + str(n_hidden2_M2) + ")", 
                str(activate_M2).split(" ")[1], round(runtime_M2,2), 
                round(acc_trainM2,3), round(accuracy_M2, 3)],
                [2, "(" + str(n_hidden1_M3) + "," + str(n_hidden2_M3) + ")", 
                str(activate_M3).split(" ")[1], round(runtime_M3,2), 
                round(acc_trainM3,3), round(accuracy_M3, 3)],
                [2, "(" + str(n_hidden1_M4) + "," + str(n_hidden2_M4) + ")", 
                str(activate_M4).split(" ")[1], round(runtime_M4,2), 
                round(acc_trainM4,3), round(accuracy_M4, 3)],
                [2, "(" + str(n_hidden1_M5) + "," + str(n_hidden2_M5) + ")", 
                str(activate_M5).split(" ")[1], round(runtime_M5,2), 
                round(acc_trainM5,3), round(accuracy_M5, 3)],
                [2, "(" + str(n_hidden1_M6) + "," + str(n_hidden2_M6) + ")", 
                str(activate_M6).split(" ")[1], round(runtime_M6,2), 
                round(acc_trainM6,3), round(accuracy_M6, 3)],
                [5, "(" + str(n_hidden1_M7) + "," + str(n_hidden2_M7) 
                + "," + str(n_hidden3_M7) + "," + str(n_hidden4_M7) 
                + "," + str(n_hidden5_M7) + ")", 
                str(activate_M7).split(" ")[1], round(runtime_M7,2), 
                round(acc_trainM7,3), round(accuracy_M7, 3)],
                [5, "(" + str(n_hidden1_M8) + "," + str(n_hidden2_M8) 
                + "," + str(n_hidden3_M8) + "," + str(n_hidden4_M8) 
                + "," + str(n_hidden5_M8) + ")", 
                str(activate_M8).split(" ")[1], round(runtime_M8,2), 
                round(acc_trainM8,3), round(accuracy_M8, 3)],
                [5, "(" + str(n_hidden1_M9) + "," + str(n_hidden2_M9) 
                + "," + str(n_hidden3_M9) + "," + str(n_hidden4_M9) 
                + "," + str(n_hidden5_M9) + ")", 
                str(activate_M9).split(" ")[1], round(runtime_M9,2), 
                round(acc_trainM9,3), round(accuracy_M9, 3)],
                [5, "(" + str(n_hidden1_M10) + "," + str(n_hidden2_M10) 
                + "," + str(n_hidden3_M10) + "," + str(n_hidden4_M10) 
                + "," + str(n_hidden5_M10) + ")", 
                str(activate_M10).split(" ")[1], round(runtime_M10,2), 
                round(acc_trainM10,3), round(accuracy_M10, 3)],
                [5, "(" + str(n_hidden1_M11) + "," + str(n_hidden2_M11) 
                + "," + str(n_hidden3_M11) + "," + str(n_hidden4_M11) 
                + "," + str(n_hidden5_M11) + ")", 
                str(activate_M11).split(" ")[1], round(runtime_M11,2), 
                round(acc_trainM11,3), round(accuracy_M11, 3)],
                [5, "(" + str(n_hidden1_M12) + "," + str(n_hidden2_M12) 
                + "," + str(n_hidden3_M12) + "," + str(n_hidden4_M12) 
                + "," + str(n_hidden5_M12) + ")", 
                str(activate_M12).split(" ")[1], round(runtime_M12,2), 
                round(acc_trainM12,3), round(accuracy_M12, 3)]]

table = tabulate(table_vals, headers=col_labels)
                                
print(table)

