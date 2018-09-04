from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint as pp
import csv
import os
import time

tf.set_random_seed(777)  # reproducibility

nb_classes = 16  # 1:punch_l 2:punch_r 3:punch_l2 4:punch_r2 5:hold

X = tf.placeholder(tf.float32, [None, 864])
Y = tf.placeholder(tf.int32, [None, nb_classes])  # 1:punch_l 2:punch_r 3:punch_l2 4:punch_r2 5:hold



#Y_one_hot = tf.one_hot(y_data, nb_classes)  # one hot
#pre = np.array(Y_one_hot, dtype=np.float32)
#print("one_hot", Y_one_hot)
#Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
#print("reshape", Y_one_hot)
#print(x_data.shape, one_hot_targets.shape)

W = tf.Variable(tf.random_normal([864, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# parameters
learning_rate = 0.001
training_epochs = 16
batch_size = 5
total_batch = int(640 / batch_size)

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
W1 = tf.get_variable("W1", shape=[864, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
W5 = tf.get_variable("W5", shape=[512, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L1, W5) + b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)

dataX = [[0 for i in range(864)]]

def python_init():
	global sess
	sess.run(init)

	saver = tf.train.Saver()
	saver.restore(sess, os.getcwd() + "/model-softmax3.ckpt")
	print(os.getcwd() + "/model-softmax3.ckpt")

'''def action_classification(arg):
	data = np.array(arg, dtype=np.float)
	print(data)
	data = data.reshape(1, 16, 54)
	global sess
	global prediction
	global pre_data
	#predict_output = sess.run(prediction, feed_dict={X: data})
	#print("Prediction:", predict_output)
	return predict_output'''

def action_classification(arg):
	global sess
	global dataX
	line = arg.split(',')
	linetodata = list(line)
	c = 0
	for a in range(864):
		dataX[0][a] = linetodata[c]
		c = c + 1
	data = np.array(dataX, dtype=np.float32)
	result = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: data, keep_prob: 1})
	if(result == 0) :
		print("Left Punch1")
	elif(result == 1):
		print("Right Punch1")
	elif(result == 2):
		print("Left Punch2")
	elif(result == 3):
		print("Right Punch2")
	elif(result == 4):
		print("Hold")
	elif(result == 5):
		print("Hello Out")
	elif(result == 6):
		print("Jump")
	elif(result == 7):
		print("Left Walk")
	elif(result == 8):
		print("Left Walk2")
	elif(result == 9):
		print("Right Walk")
	elif(result == 10):
		print("Right Walk2")
	elif(result == 11):
		print("Left Kick")
	elif(result == 12):
		print("Left Kick2")
	elif(result == 13):
		print("Right Kick")
	elif(result == 14):
		print("Right Kick2")
	elif(result == 15):
		print("Hello In")


#	    print("Prediction: {}".format(p))


#def action_classification(arg):
#	print("Enter")
#	global sess
#	global prediction
#        
#	dataX = [[[0 for rows in range(54)]for cols in range(16)]]
#	c = 0;
#	strline = hello.readline()
#	line = strline.split(',')
#	linetodata = list(line)
#	print(linetodata)
#	for a in range(16):
#		for b in range(54):
#			dataX[0][a][b] = linetodata[c]
#			c = c + 1
#	data = np.array(dataX, dtype=np.float32)
#	predict_output = sess.run(prediction, feed_dict={X: pre_data})
#	print("Prediction:", predict_output)
#	return predict_output

def python_close():
	global sess
	sess.close()


#pp.pprint(dataX)

#pp.pprint(tf.shape(x_data))

#cell = rnn.BasicLSTMCell(num_units=2, state_is_tuple=True)
#outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
#pp.pprint(outputs.eval())
