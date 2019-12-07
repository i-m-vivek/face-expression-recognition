# Author: Vivek Mittal
# https://github.com/i-m-vivek

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from sklearn.preprocessing import OneHotEncoder 

tf.set_random_seed(1)
np.random.seed(1)

def next_batch(X, y, batch_size, step):
    
    return X[step*batch_size :batch_size + step*batch_size], y[step*batch_size :batch_size + step*batch_size]

# Hyperparameter 
BATCH_SIZE = 128
LR = .001

datafile = pd.read_csv("face_recog_data.csv")
emotion_dict  = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
X = np.empty(shape = (35887, 48, 48))
y = np.array(datafile["emotion"])

for i in range(len(datafile)):
    X[i, :, :] = np.array(list(map(int, datafile["pixels"][i].split()))).reshape(48, 48)     
X = X/255

y = y.reshape(-1, 1)
onehotencoder = OneHotEncoder() 
y = onehotencoder.fit_transform(y).toarray() 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)
X_test = X_test.reshape(-1, 48, 48, 1)

#lets look on a example from our train set 
sample_img = X_train[9]
sample_label = y_train[9]
plt.imshow(sample_img, cmap ="gray")
plt.title(emotion_dict[np.argmax(sample_label)])
plt.show()

#making the placeholders for out data 
tf_x = tf.placeholder(dtype=tf.float32, shape = [None, 48, 48, 1])
#tf_x = tf.reshape(image, [-1, 48, 48, 1])
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 7])

#making our conv network 
#Input : (Batch_Size, 48, 48, 1)
conv1 = tf.layers.conv2d(tf_x, 
                         filters = 32, 
                         strides=1,
                         kernel_size=5,
                         kernel_initializer=tf.initializers.truncated_normal(seed = 42),
                         padding="same",
                         activation = tf.nn.relu)
#output: (48, 48, 32)
pool1 = tf.layers.max_pooling2d(conv1, 
                            strides=2,
                            pool_size=2)
#output: (32, 24, 24, 32)
conv2 = tf.layers.conv2d(pool1, 
                         filters = 64, 
                         strides=1,
                         kernel_size=5,
                         kernel_initializer=tf.initializers.truncated_normal(seed = 42),
                         padding="same",
                         activation = tf.nn.relu)
#output: (24, 24, 64)
pool2 = tf.layers.max_pooling2d(conv2, 
                            strides=2,
                            pool_size=2)
#output: (12, 12, 64)
conv3 = tf.layers.conv2d(pool2, 
                         filters = 128, 
                         strides=1,
                         kernel_size=5,
                         kernel_initializer=tf.initializers.truncated_normal(seed = 42),
                         padding="same",
                         activation = tf.nn.relu)
#output (12, 12, 128)
pool3 = tf.layers.max_pooling2d(conv3, 
                            strides=2,
                            pool_size=2)
#output (6, 6, 128)
flat = tf.reshape(pool3, shape=[-1, 128*6*6])
dense1 = tf.layers.dense(flat, 512,activation = tf.nn.relu, kernel_initializer=tf.initializers.glorot_normal(seed = 42))
dense2 = tf.layers.dense(dense1, 512,activation = tf.nn.relu, kernel_initializer=tf.initializers.glorot_normal(seed = 42))
output = tf.layers.dense(dense2, 7)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits = output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy= tf.metrics.accuracy(labels=tf.math.argmax(tf_y, axis = 1), predictions=tf.math.argmax(output,axis = 1))[1] #creates 2 local variables 

# defining the session
sess = tf.Session()
# Initialize all the params 
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
# running the session
sess.run(init_op) 

num_batches = X_train.shape[0]//BATCH_SIZE
for epoch in range(50):
#    print("============Check 1===============")
    for batch in range(num_batches - 1):
#        print("++++++++++++Check2+++++++")
        temp, b_y = next_batch(X_train, y_train, 128, batch)
        b_x = temp.reshape(-1, 48, 48, 1)
#        print("++++++++++++Check3+++++++")
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
#        print("++++++++++++Check4+++++++")
        
        if epoch%5 == 0:
            accuracy_, flat_repr = sess.run([accuracy, flat], {tf_x: X_test, tf_y: y_test})
            print('Step:', epoch, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

    
    
