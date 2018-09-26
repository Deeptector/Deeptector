from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint as pp
import csv
import os, sys
import time
import gc
import socket
from socket import *
from select import select

gc.disable()

tf.set_random_seed(777)

nb_classes = 18  

X = tf.placeholder(tf.float32, [None, 864])
Y = tf.placeholder(tf.int32, [None, nb_classes])  

W = tf.Variable(tf.random_normal([864, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# parameters
learning_rate = 0.001
training_epochs = 18
batch_size = 5
total_batch = int(2142 / batch_size)

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

count1 = 0
count2 = -10

punch1 = 0
punch2 = 0

hold1 = 0
hold2 = 0

kick1 = 0
kick2 = 0

clap1 = 0
clap2 = 0

jump1 = 0
jump2 = 0

walk1 = 0
walk2 = 0

detect = 0

def python_init():
	global sess
	sess.run(init)
	global clientSocket
	global ADDR
	saver = tf.train.Saver()
	saver.restore(sess, os.getcwd() + "/model-deeptector.ckpt")

def action_classification(arg):
	global sess
	global dataX
	line = arg.split(',')
	linetodata = list(line)
	c = 0
	global count1 
	global count2

	global punch1
	global punch2

	global hold1
	global hold2

	global kick1
	global kick2

	global clap1
	global clap2

	global jump1
	global jump2

	global walk1
	global walk2

	global clientSocket

	for a in range(864):
		dataX[0][a] = linetodata[c]
		c += 1
	data = np.array(dataX, dtype=np.float32)
	result = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: data, keep_prob: 1})
	if(result == 0) :
#		print("Left Punch1") # 왼팔 오른쪽 Punch_L1
		punch1 += 1
		punch2 += 1
	elif(result == 1):
#		print("Right Punch1") # 오른팔 왼쪽 Punch_R1
		punch1 += 1
		punch2 += 1
	elif(result == 2):
#		print("Left Punch2") # 왼팔 왼쪽 Punch_L2
		punch1 += 1
		punch2 += 1
	elif(result == 3):
#		print("Right Punch2") # 오른팔 오른쪽 Punch_R2
		punch1 += 1
		punch2 += 1
	elif(result == 4):
#		print("Hold")
		hold1 += 1
		hold2 += 1
	elif(result == 5):
#		print("Jump")
		jump1 += 1
		jump2 += 1
	elif(result == 6):
#		print("Left Walk") # 왼쪽으로 걷기 walk_left
		walk1 += 1
		walk2 += 1
	elif(result == 7):
#		print("Left Walk2") # 왼쪽 대각선으로 걷기 walk_left2
		walk1 += 1
		walk2 += 1
	elif(result == 8):
#		print("Right Walk") # 오른쪽으로 걷기 walk_right
		walk1 += 1
		walk2 += 1
	elif(result == 9):
#		print("Right Walk2") # 오른쪽 대각선으로 걷 walk_right2
		walk1 += 1
		walk2 += 1
	elif(result == 10):
#		print("Left Kick1") # 오른다리로 왼쪽 때리기 kick_letf_in
		kick1 += 1
		kick2 += 1
	elif(result == 11):
#		print("Left Kick2") # 왼다리로 왼쪽 때리기 kick_left_out
		kick1 += 1
		kick2 += 1
	elif(result == 12):
#		print("Right Kick1") # 왼다리로 오른쪽 때리기 kick_right_in
		kick1 += 1
		kick2 += 1
	elif(result == 13):
#		print("Right Kick2") # 오른다리로 오른쪽 때리기 kick_right_out
		kick1 += 1
		kick2 += 1
	elif(result == 14):
#		print("Left Kick") # 오른다리로 왼쪽 때리기 (승 left_kick
		kick1 += 1
		kick2 += 1
	elif(result == 15):
#		print("Right Kick") # 오른다리로 오른쪽 때리기 (승 right_kick
		kick1 += 1
		kick2 += 1
	elif(result == 16):
#		print("Clap1")
		clap1 += 1
		clap2 += 1
	elif(result == 17):
#		print("Clap2")
		clap1 += 1
		clap2 += 1
	count1 += 1
	count2 += 1

	if(count1 == 20):
		a = [punch1, hold1, kick1, clap1, jump1, walk1]
		maxIdx = a.index(max(a))
		if(maxIdx == 0) :
			print("punch")
		elif(maxIdx == 1) :
			print("hold")
		elif(maxIdx == 2) :
			print("kick")
		elif(maxIdx == 3) :
			print("clap")
		elif(maxIdx == 4) :
			print("jump")
		elif(maxIdx == 5) :
			print("walk")
		count1 = 0
		punch1 = 0
		hold1 = 0
		kick1 = 0
		clap1 = 0
		jump1 = 0
		walk1 = 0

	elif(count2 == 20):
		a = [punch2, hold2, kick2, clap2, jump2, walk2]
		maxIdx = a.index(max(a))
		if(maxIdx == 0) :
			print("punch")
		elif(maxIdx == 1) :
			print("hold")
		elif(maxIdx == 2) :
			print("kick")
		elif(maxIdx == 3) :
			print("clap")
		elif(maxIdx == 4) :
			print("jump")
		elif(maxIdx == 5) :
			print("walk")
		count2 = 0
		punch2 = 0
		hold2 = 0
		kick2 = 0
		clap2 = 0
		jump2 = 0
		walk2 = 0


def python_close():
	global sess
	sess.close()

