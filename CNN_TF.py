# Code Inspired from https://github.com/MorvanZhou/Tensorflow-Tutorial
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

tf.set_random_seed(1)
np.random.seed(1)

# Hyperparamters 
BATCH_SIZE = 50
LR = .001

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist", one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)

# Look On a sample image from the train set 
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap = "gray")
plt.title("%i" %np.argmax(mnist.train.labels[0]))
plt.show()

# Defining the inputs and the outputs for our model 
# None -> for the batch_size param
tf_x = tf.placeholder(tf.float32, [None, 28*28])/255
image = tf.reshape(tf_x, [-1, 28, 28, 1]) # Batch Size, Height, Width, Channels
# tf_y -> stores the labels a 10d vector 
tf_y = tf.placeholder(tf.int32, [None, 10]) #y


# Making the CNN Model 

# Input: (28, 28, 1)
conv1 = tf.layers.conv2d(image,
                         filters=16, kernel_size=5, 
                         strides=1, padding="same", 
                         activation=tf.nn.relu)
#Input (28, 28, 16)
pool1 = tf.layers.max_pooling2d(conv1, 
                                pool_size=2,
                                strides=2)
#Input (14, 14, 16)
conv2 = tf.layers.conv2d(pool1,
                         filters=32,
                         strides=1,
                         kernel_size=5,
                         padding="same",
                         activation=tf.nn.relu)
#Input: (14,14,32)
pool2 = tf.layers.max_pooling2d(conv2,
                                pool_size=2,
                                strides=2)
#input: (7, 7, 32)
flat = tf.reshape(pool2, [-1, 7*7*32]) 
# Output Layer from which we recieve the final output 
output  = tf.layers.dense(flat, 10)

# as our data is a multi categorical data softmax_cross_entropy will be the best 
loss = tf.losses.softmax_cross_entropy(onehot_labels= tf_y, logits=output)
# using adam optimizer
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# using accuracy as our metric
accuracy= tf.metrics.accuracy(labels=tf.math.argmax(tf_y, axis = 1), predictions=tf.math.argmax(output,axis = 1))[1]

# defining the session
sess = tf.Session()
# Initialize all the params 
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
# running the session
sess.run(init_op) 


from matplotlib import cm

def plot_with_labels (lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X,Y,labels):
        c = cm.rainbow(int(225*s/9))
        plt.text(x, y, s, backgroundcolor = c, fontsize = 9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("Visualize the last layer")
    plt.show()
    plt.pause(.01)
    
plt.ion()
for step in range(600):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    
    if step%50 == 0:
        accuracy_, flat_repr = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
        # plotting the result in 2d to see the cluster formation for particular digit 
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
        low_dim_embs = tsne.fit_transform(flat_repr[:plot_only, :])
        labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
plt.ioff()

# Testing our model model
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')


