# MLP framework for fashion mnist dataset clothes classification
import tensorflow as tf
from tensorflow import keras

sample = tf.keras.datasets.fashion_mnist.load_data()

(X_train_full, y_train_full), (X_test, y_test) = sample
print(X_train_full.dtype, X_train_full.shape)

# Data selection for validation
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Creating MLP model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28])) #input layer
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dropout(0.2))  # Drops 20% of neurons during training to reduce overfitting
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation = "softmax"))  # "softmax" activation function because the calsses are discrete variables

# model's layers summary
model.summary()

# Compiling the model (sgd = Stochastic Gradient Descent)
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer = "sgd",  
              metrics = ["accuracy"])  # accuracy rate

# training the model
history = model.fit(X_train, y_train, epochs = 50,
                    validation_data = (X_valid, y_valid))

# Testing the model with unseen data to evaluaate the predictions generalization error
model.evaluate(X_test/255.0, y_test)

#Learning curves
import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# Probabilities for each class
y_pred_prob = model.predict(x_test / 255.0)
y_pred = np.argmax(y_pred_prob, axis=1) # converting to the most probable class

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')

plt.xlabel('Predictions')
plt.ylabel('Real Values')
plt.title('Confusion Matrix - MNIST')
plt.show()
