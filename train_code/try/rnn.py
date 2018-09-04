from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint as pp
import csv
import os
import time

tf.set_random_seed(777)  # reproducibility
num_classes = 1
input_dim = 54  # data_size
hidden_size = 10  # output from the LSTM
batch_size = 1   # one sentence
sequence_length = 16  # |ihello| == 6
learning_rate = 0.1

dataY = [[[1]],[[2]],[[3]],[[4]],[[5]]]

y1_data = np.array(dataY[0], dtype=np.float32)
y2_data = np.array(dataY[1], dtype=np.float32)
y3_data = np.array(dataY[2], dtype=np.float32)
y4_data = np.array(dataY[3], dtype=np.float32)
y5_data = np.array(dataY[4], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.float32, [None, 1])  # Y label

def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(3)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], num_classes, activation_fn=None)

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
prediction = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - prediction)))

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)

def python_init():
	#config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.3
	global sess
	sess.run(init)

	saver = tf.train.Saver()
	saver.restore(sess, os.getcwd() + "/model2.ckpt")
	print(os.getcwd() + "/model2.ckpt")

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
	global prediction
	global Y_pred
	dataX = [[[0 for rows in range(54)]for cols in range(16)]]
	line = arg.split(',')
	linetodata = list(line)
	c = 0
	for a in range(16):
		for b in range(54):
			dataX[0][a][b] = linetodata[c]
			c = c + 1
	data = np.array(dataX, dtype=np.float32)
	predict_output = sess.run(Y_pred, feed_dict={X: data})
	if 0.8 < predict_output and 1.2 > predict_output:
		print("person 1 : punch_l")
	elif 1.8 < predict_output and 2.2 > predict_output:
		print("person 1 : punch_r")
	elif 2.8 < predict_output and 3.2 > predict_output:
		print("person 1 : punch_l2")
	elif 3.8 < predict_output and 4.2 > predict_output:
		print("person 1 : punch_r2")
	elif 4.8 < predict_output and 5.2 > predict_output:
		print("person 1 : Hold")
	elif 5.8 < predict_output and 6.2 > predict_output:
		print("person 1 : walk_l")
	elif 6.8 < predict_output and 7.2 > predict_output:
		print("person 1 : walk_r")
	elif 7.8 < predict_output and 8.2 > predict_output:
		print("person 1 : walk_l2")
	elif 8.8 < predict_output and 9.2 > predict_output:
		print("person 1 : walk_r2")
	print("output :", predict_output)
	return predict_output 


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

#python_init()
#action_classification()
#python_close()

#pp.pprint(dataX)

#pp.pprint(tf.shape(x_data))

#cell = rnn.BasicLSTMCell(num_units=2, state_is_tuple=True)
#outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
#pp.pprint(outputs.eval())
