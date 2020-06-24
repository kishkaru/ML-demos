import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# "Hello world" dataset: digits 0-9, 10 28x28px images
mnist = tf.keras.datasets.mnist

# x_train: features (pixel values of digits)
# y_train: labels (0-9)
# x_test: out-of-sample data (features)
# y_test: out-of-sample data (labels)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Peek at dataset
# print(y_train[0])
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()

# Normalize features data to values between [0, 1]
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# A feed forward model (go in forward order)
model = tf.keras.models.Sequential()

# Input layer: flatten 28x28px into 1x784px
model.add(tf.keras.layers.Flatten())
# Hidden layer 1:
# Dense: 1D "fully connected" layer where each node connects to each prior and next node.
# 128 nodes
# rectified linear (relu) activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Hidden layer 2: "fully connected" layer where each node connects to each prior and next node
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Output layer: 10 nodes (1 per each output prediction [0,9])
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Compile model to optimize
# optimizer: default optimizer
# loss: loss function to minimize
# metrics: list of metrics to monitor
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
# epochs: episodes/iterations over dataset
history = model.fit(x_train, y_train, epochs=3)
print(history.history)
model.save('epic_num_reader.model')

# Test on out-of-sample data to see if model generalized or overfit (memorized)
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

# Make predictions based on model
predictions = model.predict(x_test)
print(np.argmax(predictions[0]))
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()
