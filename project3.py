import numpy as np
import tensorflow as tf

# Load MNIST data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# Load training data and test data
train_data = mnist.train.images # Size (55000, 785)
eval_data = mnist.test.images # Size (10000,7 84)

print(np.shape(train_data))
print(np.shape(eval_data))
