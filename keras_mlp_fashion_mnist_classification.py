# MLP framework for fashion mnist dataset clothes classification
import tensorflow as tf
from tensorflow import keras

sample = tf.keras.datasets.fashion_mnist.load_data()

(x_train_full, y_train_full), (x_test, y_test) = sample
print(x_train_full.dtype, x_train_full.shape)

# Data selection for validation
x_valid, x_train = x_train_full[:5000]/255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Creating MLP model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28])) #input layer
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dropout(0.2))  # Drops 20% of neurons during training to reduce overfitting
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation = "softmax"))  # "softmax" activation function because the calsses are discrete variables

model.summary()

# Compiling the model (sgd = Stochastic Gradient Descent)
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer = "sgd",  
              metrics = ["accuracy"])  # accuracy rate

# training the model
history = model.fit(x_train, y_train, epochs = 50,
                    validation_data = (x_valid, y_valid))

# Testing the model with unseen data to evaluaate the predictions generalization error
model.evaluate(x_test/255.0, y_test)

#Learning curves
import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
