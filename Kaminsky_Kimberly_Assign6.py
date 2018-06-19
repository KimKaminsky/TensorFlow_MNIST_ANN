# Kimberly Kaminsky - Assignment #6
# Neural Networks

####################
# Import Libraries #
####################

# import base packages into the namespace for this program
import warnings
import numpy as np
import os
import sys
import time

from astropy.table import Table
from tabulate import tabulate

# Use to build neural network
import tensorflow as tf

# Stores MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

# TensorBoard graphs
from IPython.display import clear_output, Image, display, HTML


#################### 
# Define constants #
####################

N_INPUTS = 28*28  # MNIST dataset features
N_OUTPUTS = 10    # Categories (number of digits)
DATAPATH = os.path.join("D:/","Kim MSPA", "Predict 422", "Assignments", "Assignment6", "")

#############
# Functions #
#############

# function to clear output console
def clear():
    print("\033[H\033[J")
    
# reset graph to make output stable across runs
def reset_graph(seed=111):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)  

def two_layer_NN(n_hidden1, n_hidden2, activate):
    # reset graph
    reset_graph()
    
    tf.set_random_seed(111)
    
    # setup placeholder nodes to represent the training data and targets
    X = tf.placeholder(tf.float32, shape=(None, N_INPUTS), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y") 
    
    
    # Create Neural Network
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                                activation=activate)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                                activation=activate)
        logits = tf.layers.dense(hidden2, N_OUTPUTS, name="outputs") 
    
    # Cost function used to train Neural Network
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, 
                                                                logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    
    # Create optimizer to tweak model paramtersto minimize the cost function
    learning_rate = 0.01
    
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
        
    # Measure classification performance
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  
    
    # Node to initialize all the variables
    init = tf.global_variables_initializer()
    
    # Create saver object to save traned model parameters to disk
    saver = tf.train.Saver(save_relative_paths=True) 

    # Execute Model 
    
    # Train model 
    n_epochs = 20
    batch_size = 50
    
    # Start clock to time training time for NN
    start_time = time.clock()
    
    # The next_batch function is essentially doing random sampling so the results
    # won't be consistent from run to run
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
    
    # Now restore graph parameters and run against holdout data for final score      
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        accuracy = accuracy.eval(feed_dict={X: mnist.test.images, 
                                        y: mnist.test.labels})
    
    # Start clock to time training time for NN
    stop_time = time.clock()
    
    #Total Time
    runtime = stop_time - start_time  
    
    return accuracy, runtime, acc_test
    
def five_layer_NN(n_hidden1, n_hidden2, n_hidden3, n_hidden4,
                    n_hidden5, activate):
    # reset graph
    reset_graph()
    
    tf.set_random_seed(111)
    
    # setup placeholder nodes to represent the training data and targets
    X = tf.placeholder(tf.float32, shape=(None, N_INPUTS), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y") 
    
    
    # Create Neural Network
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
    
    # Cost function used to train Neural Network
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, 
                                                                logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    
    # Create optimizer to tweak model paramtersto minimize the cost function
    learning_rate = 0.01
    
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
        
    # Measure classification performance
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  
    
    # Node to initialize all the variables
    init = tf.global_variables_initializer()
    
    # Create saver object to save traned model parameters to disk
    saver = tf.train.Saver(save_relative_paths=True) 

    # Execute Model 
    
    # Train model 
    n_epochs = 20
    batch_size = 50
    
    # Start clock to time training time for NN
    start_time = time.clock()
    
    # The next_batch function is essentially doing random sampling so the results
    # won't be consistent from run to run
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
    
    # Now restore graph parameters and run against holdout data for final score      
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        accuracy = accuracy.eval(feed_dict={X: mnist.test.images, 
                                        y: mnist.test.labels})
    
    # Start clock to time training time for NN
    stop_time = time.clock()
    
    #Total Time
    runtime = stop_time - start_time  
    
    return accuracy, runtime, acc_test

     
######################
# General Prep Work #
######################

# Turn off future code warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)


####################
# Data Preparation #
####################

# Read in dataset
mnist = input_data.read_data_sets("/tmp/data/")

# Take a look at the shape of the train,validate and test setss
X_train = mnist.train.images
X_validate = mnist.validation.images
X_test = mnist.test.images

print("Shape of Training data: ", X_train.shape)
print("Shape of Validate data: ", X_validate.shape)
print("Shape of Test data: ", X_test.shape)


#########################
# Neural Network Models #
#########################

# Building several models to assess classification performance accuracy
# and processing time 

######################
# Construct Model 1: #
#   2 layer          #
#   hidden (300,100) #
#   activation: reLU #
######################

# Basic NN structure
n_hidden1_M1 = 300
n_hidden2_M1 = 100
activate_M1 = tf.nn.relu
 
accuracy_M1, runtime_M1, acc_trainM1 = two_layer_NN(n_hidden1_M1, n_hidden2_M1, 
                                        activate_M1)

######################
# Construct Model 2 :#
#   2 layer          #
#   hidden (200,200) #
#   activation: reLU #
######################

# Basic NN structure
n_hidden1_M2 = 200
n_hidden2_M2 = 200
activate_M2 = tf.nn.relu

accuracy_M2, runtime_M2, acc_trainM2 = two_layer_NN(n_hidden1_M2, n_hidden2_M2, 
                                        activate_M2)


######################
# Construct Model 3 :#
#   2 layer          #
#   hidden (300,100) #
#   activation: tanh #
######################

# Basic NN structure
n_hidden1_M3 = 300
n_hidden2_M3 = 100
activate_M3 = tf.nn.tanh

accuracy_M3, runtime_M3, acc_trainM3 = two_layer_NN(n_hidden1_M3, n_hidden2_M3,
                                        activate_M3)


######################
# Construct Model 4 :#
#   2 layer          #
#   hidden (200,200) #
#   activation: tanh #
######################

# Basic NN structure
n_hidden1_M4 = 200
n_hidden2_M4 = 200
activate_M4 = tf.nn.tanh

accuracy_M4, runtime_M4, acc_trainM4 = two_layer_NN(n_hidden1_M4, n_hidden2_M4, 
                                        activate_M4)

######################
# Construct Model 5 :#
#   2 layer          #
#   hidden (300,100) #
#   activation: Elu #
######################

# Basic NN structure
n_hidden1_M5 = 300
n_hidden2_M5 = 100
activate_M5 = tf.nn.elu

accuracy_M5, runtime_M5, acc_trainM5 = two_layer_NN(n_hidden1_M5, n_hidden2_M5, 
                                        activate_M5)

######################
# Construct Model 6 :#
#   2 layer          #
#   hidden (200,200) #
#   activation: Elu  #
######################

# Basic NN structure
n_hidden1_M6 = 200
n_hidden2_M6 = 200
activate_M6 = tf.nn.elu

accuracy_M6, runtime_M6, acc_trainM6 = two_layer_NN(n_hidden1_M6, n_hidden2_M6, 
                                        activate_M6)

##############################
# Construct Model 7 :        #
#   5 layer                  #
#   hidden (80,80,80,80,80)  #
#   activation: relu         #
##############################

# Basic NN structure
n_hidden1_M7 = 80
n_hidden2_M7 = 80
n_hidden3_M7 = 80
n_hidden4_M7 = 80
n_hidden5_M7 = 80
activate_M7 = tf.nn.relu

accuracy_M7, runtime_M7, acc_trainM7 = five_layer_NN(n_hidden1_M7, n_hidden2_M7, 
                        n_hidden3_M7, n_hidden4_M7, n_hidden5_M7, activate_M7)
                        
                        
##############################
# Construct Model 8:         #
#   5 layer                  #
#   hidden (160,100,80,40,20)#
#   activation: relu         #
##############################

# Basic NN structure
n_hidden1_M8 = 160
n_hidden2_M8 = 100
n_hidden3_M8 = 80
n_hidden4_M8 = 40
n_hidden5_M8 = 20
activate_M8 = tf.nn.relu

accuracy_M8, runtime_M8, acc_trainM8 = five_layer_NN(n_hidden1_M8, n_hidden2_M8, 
                        n_hidden3_M8, n_hidden4_M8, n_hidden5_M8, activate_M8)

##############################
# Construct Model 9 :        #
#   5 layer                  #
#   hidden (80,80,80,80,80)  #
#   activation:  elu         #
##############################

# Basic NN structure
n_hidden1_M9 = 80
n_hidden2_M9 = 80
n_hidden3_M9 = 80
n_hidden4_M9 = 80
n_hidden5_M9 = 80
activate_M9 = tf.nn.elu

accuracy_M9, runtime_M9, acc_trainM9 = five_layer_NN(n_hidden1_M9, n_hidden2_M9, 
                        n_hidden3_M9, n_hidden4_M9, n_hidden5_M9, activate_M9)
                        
##############################
# Construct Model 10:        #
#   5 layer                  #
#   hidden (160,100,80,40,20)#
#   activation:  elu         #
##############################

# Basic NN structure
n_hidden1_M10 = 160
n_hidden2_M10 = 100
n_hidden3_M10 = 80
n_hidden4_M10 = 40
n_hidden5_M10 = 20
activate_M10 = tf.nn.elu

accuracy_M10, runtime_M10, acc_trainM10 = five_layer_NN(n_hidden1_M10, 
                            n_hidden2_M10, n_hidden3_M10, n_hidden4_M10, 
                            n_hidden5_M10, activate_M10)
                
#################################
# Construct Model 11:           #
#   5 layer                     #
#   hidden (156,156,156,156,156)#
#   activation:  elu            #
#################################

# Basic NN structure
n_hidden1_M11 = 156
n_hidden2_M11 = 156
n_hidden3_M11 = 156
n_hidden4_M11 = 156
n_hidden5_M11 = 156
activate_M11 = tf.nn.elu

accuracy_M11, runtime_M11, acc_trainM11 = five_layer_NN(n_hidden1_M11, 
                            n_hidden2_M11, n_hidden3_M11, n_hidden4_M11, 
                            n_hidden5_M11, activate_M11)
                
#################################
# Construct Model 12:           #
#   5 layer                     #
#   hidden (156,156,156,156,156)#
#   activation:  relu           #
#################################

# Basic NN structure
n_hidden1_M12 = 156
n_hidden2_M12 = 156
n_hidden3_M12 = 156
n_hidden4_M12 = 156
n_hidden5_M12 = 156
activate_M12 = tf.nn.relu

accuracy_M12, runtime_M12, acc_trainM12 = five_layer_NN(n_hidden1_M12, 
                            n_hidden2_M12, n_hidden3_M12, n_hidden4_M12, 
                            n_hidden5_M12, activate_M12)
                
#################################
# Construct Model 13:           #
#   5 layer                     #
#   hidden (350,200,100,80,50)  #
#   activation:  elu            #
#################################

# Basic NN structure
n_hidden1_M13 = 350
n_hidden2_M13 = 200
n_hidden3_M13 = 100
n_hidden4_M13 = 80
n_hidden5_M13 = 50
activate_M13 = tf.nn.elu

accuracy_M13, runtime_M13, acc_trainM13 = five_layer_NN(n_hidden1_M13, 
                            n_hidden2_M13, n_hidden3_M13, n_hidden4_M13,
                            n_hidden5_M13, activate_M13)
                
#################################
# Construct Model 14:           #
#   5 layer                     #
#   hidden (350,200,100,80,50)  #
#   activation:  relu           #
#################################

# Basic NN structure
n_hidden1_M14 = 350
n_hidden2_M14 = 200
n_hidden3_M14 = 100
n_hidden4_M14 = 80
n_hidden5_M14 = 50
activate_M14 = tf.nn.relu

accuracy_M14, runtime_M14, acc_trainM14 = five_layer_NN(n_hidden1_M14, 
                            n_hidden2_M14, n_hidden3_M14, n_hidden4_M14, 
                            n_hidden5_M14, activate_M14)                

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

print("Accuracy Score for Model 13: ", accuracy_M13)
print("Run Time for Model 13: ", runtime_M13) 

print("Accuracy Score for Model 14: ", accuracy_M14)
print("Run Time for Model 14: ", runtime_M14) 


########################
# Create Output Table  #
########################

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
                round(acc_trainM12,3), round(accuracy_M12, 3)],
                [5, "(" + str(n_hidden1_M13) + "," + str(n_hidden2_M13) 
                + "," + str(n_hidden3_M13) + "," + str(n_hidden4_M13) 
                + "," + str(n_hidden5_M13) + ")", 
                str(activate_M13).split(" ")[1], round(runtime_M13,2), 
                round(acc_trainM13,3), round(accuracy_M13, 3)],
                [5, "(" + str(n_hidden1_M14) + "," + str(n_hidden2_M14) 
                + "," + str(n_hidden3_M14) + "," + str(n_hidden4_M14) 
                + "," + str(n_hidden5_M14) + ")", 
                str(activate_M14).split(" ")[1], round(runtime_M14,2), 
                round(acc_trainM14,3), round(accuracy_M14, 3)]]

table = tabulate(table_vals, headers=col_labels)
                                
print(table)


