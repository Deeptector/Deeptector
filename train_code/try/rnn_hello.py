from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint as pp
import csv

tf.set_random_seed(777)  # reproducibility

num_classes = 16
input_dim = 54  # data_size
hidden_size = 2  # output from the LSTM
batch_size = 1   # one sentence
sequence_length = 16  # |ihello| == 6
learning_rate = 0.1

dataY = [[[1, 0, 0, 0, 0]],[[0, 1, 0, 0, 0]],[[0, 0, 1, 0, 0]],[[0, 0, 0, 1, 0]],[[0, 0, 0, 0, 1]]]

y1_data = np.array(dataY[0], dtype=np.float32)
y2_data = np.array(dataY[1], dtype=np.float32)
y3_data = np.array(dataY[2], dtype=np.float32)
y4_data = np.array(dataY[4], dtype=np.float32)
y5_data = np.array(dataY[5], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, 5])  # Y label

# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

#cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
#initial_state = cell.zero_state(batch_size, tf.float32)
#outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(3)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	count = 0
	for i in range(5000):
		test_l = open('data_test_l.csv', 'r')
		test_r = open('data_test_r.csv', 'r')
		test_l2 = open('data_test_l2.csv', 'r')
		test_r2 = open('data_test_r2.csv', 'r')
		hold = open('data_hold.csv', 'r')
		j=0
		dataX = [[[0 for rows in range(54)]for cols in range(16)]]
#		helloreader = csv.reader(hello)
#		clapreader = csv.reader(clap)
#		holdreader = csv.reader(hold)
		for j in range(1 * 5) :
			if j % 5 == 0 :
				strline = test_l.readline()
				line = strline.split(',')
				linetodata = list(line)
				c = 0;
				for a in range(16):
					for b in range(54):
						dataX[0][a][b] = linetodata[c]
						c = c + 1
				x1_data = np.array(dataX, dtype=np.float32)
				l, _ = sess.run([loss, train], feed_dict={X: x1_data, Y: y1_data})
				result = sess.run(prediction, feed_dict={X: x1_data})
				count = count + 1
				print(i, count, "loss:", l, "Prediction:", result)
			elif j % 5 == 1 :
				strline = test_r.readline()
				line = strline.split(',')
				linetodata = list(line)
				c = 0;
				for a in range(16):
					for b in range(54):
						dataX[0][a][b] = linetodata[c]
						c = c + 1
				x1_data = np.array(dataX, dtype=np.float32)
				l, _ = sess.run([loss, train], feed_dict={X: x1_data, Y: y2_data})
				result = sess.run(prediction, feed_dict={X: x1_data})
				count = count + 1
				print(i, count, "loss:", l, "Prediction:", result)
			elif j % 5 == 2 :
				strline = test_l2.readline()
				line = strline.split(',')
				linetodata = list(line)
				c = 0;
				for a in range(16):
					for b in range(54):
						dataX[0][a][b] = linetodata[c]
						c = c + 1
				x1_data = np.array(dataX, dtype=np.float32)
				l, _ = sess.run([loss, train], feed_dict={X: x1_data, Y: y3_data})
				result = sess.run(prediction, feed_dict={X: x1_data})
				count = count + 1
				print(i, count, "loss:", l, "Prediction:", result)
			elif j % 5 == 3 :
				strline = test_r2.readline()
				line = strline.split(',')
				linetodata = list(line)
				c = 0;
				for a in range(16):
					for b in range(54):
						dataX[0][a][b] = linetodata[c]
						c = c + 1
				x1_data = np.array(dataX, dtype=np.float32)
				l, _ = sess.run([loss, train], feed_dict={X: x1_data, Y: y4_data})
				result = sess.run(prediction, feed_dict={X: x1_data})
				count = count + 1
				print(i, count, "loss:", l, "Prediction:", result)
			else :
				strline = hold.readline()
				line = strline.split(',')
				linetodata = list(line)
				c = 0;
				for a in range(16):
					for b in range(54):
						dataX[0][a][b] = linetodata[c]
						c = c + 1
				x1_data = np.array(dataX, dtype=np.float32)
				l, _ = sess.run([loss, train], feed_dict={X: x1_data, Y: y3_data})
				result = sess.run(prediction, feed_dict={X: x1_data})
				count = count + 1
				print(i, count, "loss:", l, "Prediction:", result)
	saver.save(sess, "trained_model", global_step=5000)

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
