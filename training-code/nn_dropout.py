# Lab 10 MNIST and Dropout
import tensorflow as tf
import numpy as np
import random

# import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility

xy = np.loadtxt('train_example.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, -1]

#output의 갯수!!!!!!
nb_classes = 18  # 1:punch_l 2:punch_r 3:punch_l2 4:punch_r2 5:hold

X = tf.placeholder(tf.float32, [None, 864])
Y = tf.placeholder(tf.int32, [None, nb_classes])  # 1:punch_l 2:punch_r 3:punch_l2 4:punch_r2 5:hold

y_data = y_data.astype(int)
one_hot_targets = np.eye(nb_classes)[y_data]
print(one_hot_targets)


W = tf.Variable(tf.random_normal([864, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# parameters
learning_rate = 0.0001
training_epochs = 40
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

W5 = tf.get_variable("W5", shape=[512, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L1, W5) + b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
# train my model
for epoch in range(50):
    avg_cost = 0
    for i in range(2142):
        feed_dict = {X: x_data, Y: one_hot_targets, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))
saver.save(sess, 'model-deeptector.ckpt')
print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: x_data, Y: one_hot_targets, keep_prob: 1}))
a=2142
b=0
'''
# Get one and predict
for i in range(2142):
	print("Label: ", sess.run(tf.argmax(one_hot_targets[i:i + 1], 1)))
	result = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: x_data[i:i + 1], keep_prob: 1})
	print("Predict : ", result)
	if(sess.run(tf.argmax(one_hot_targets[i:i + 1], 1)) == result):
		b=b+1

print("Acc : ", (b/a*100))
'''

