from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint as pp
import csv
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility
num_classes = 9
input_dim = 54  # data_size
input_steps = 16
hidden_size = 128  # output from the LSTM
batch_size = 64   # one sentence
#sequence_length = 16  # |ihello| == 6
learning_rate = 0.001

'''
dataY = [[[2]],[[4]],[[6]],[[8]],[[10]],[[12]],[[14]],[[16]],[[18]]]

y1_data = np.array(dataY[0], dtype=np.float32)
y2_data = np.array(dataY[1], dtype=np.float32)
y3_data = np.array(dataY[2], dtype=np.float32)
y4_data = np.array(dataY[3], dtype=np.float32)
y5_data = np.array(dataY[4], dtype=np.float32)
y6_data = np.array(dataY[5], dtype=np.float32)
y7_data = np.array(dataY[6], dtype=np.float32)
y8_data = np.array(dataY[7], dtype=np.float32)
y9_data = np.array(dataY[8], dtype=np.float32)
'''

x = tf.traspose(x, [1, 0, 2]) #??
x = tf.reshape(x, [-1, input_dim)
x = tf.split(x, input_steps, 0)

X = tf.placeholder(tf.float32, [None, input_steps, input_dim])  # X one-hot
Y = tf.placeholder(tf.float32, [None, num_classes])  # Y label

weights = tf.Variable(tf.random_normal(hidden_size, num_classes), name='weights')
biases = tf.Variable(tf.random_normal([num_classes]), name='biases')

lstm_cell = rnn.BasicLSTMCell(hidden_size)
outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
rnn_model = tf.matmul(outputs[-1], weights) + biases # pred
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn_model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(rnn_model,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	count = 1

	saver = tf.train.Saver(tf.all_variables())

	while count * batch_size < 10000:
		
	
	
	saver.save(sess, 'model2.ckpt')

#			line = helloreader.readline()
#			linetodata = list(line)
#			c = 0;
#			for a in range(16):
#				for b in range(54):
#					dataX[0][a][b] = linetodata[c]
#					c = c + 1
#			x1_data = np.array(dataX, dtype=np.float32)
#			l, _ = sess.run([loss, train], feed_dict={X: x1_data, Y: y1_data})
#			result = sess.run(prediction, feed_dict={X: x1_data})
#			count = count + 1
#			print(i, count, "loss:", l, "Prediction:", result)
#			else :
#				l, _ = sess.run([loss, train], feed_dict={X: x2_data, Y: y2_data})
#				result = sess.run(prediction, feed_dict={X: x2_data})
#				print(i, "loss:", l, "Prediction:", result)
	
#	result = sess.run(prediction, feed_dict={X: pre_data})
#	print("Training End - Prediction X :", result)
#	result = sess.run(prediction, feed_dict={X: pre_data2})
#	print("Training End - Prediction Y :", result)

#pp.pprint(dataX)

#pp.pprint(tf.shape(x_data))

#cell = rnn.BasicLSTMCell(num_units=2, state_is_tuple=True)
#outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
#pp.pprint(outputs.eval())
