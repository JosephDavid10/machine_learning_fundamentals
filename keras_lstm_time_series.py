import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 50,1000)
data = np.sin(t)

# Function to create sliding windows for the RNN
def sliding_window(data, window=50):
    X, y= [], [] 

    for i in range(len(data) - window): 
        X.append(data[i:i+window]) 
        y.append(data[i+window])

    return np.array(X), np.array(y) 

X, y = sliding_window(data)

X = X.reshape((X.shape[0], X.shape[1], 1))

model_rnn = keras.models.Sequential([
    keras.layers.LSTM(64, input_shape=(50,1), activation='tanh'), # tanh is suggested for RNNs
    keras.layers.Dense(1)])

model_rnn.compile(optimizer = 'adam', loss='mse') #Regression problem

model_rnn.fit(X,y, epochs=10, batch_size=32)
last_window = X[-1].reshape(1,50,1)
predict = model_rnn.predict(last_window)
print(f'Next real value: {y[-1]}')
print(f'RNN prediction: {predict[0][0]}')

predictions = model_rnn.predict(X)
plt.plot(y, label='Real')
plt.plot(predictions, label='Prediction')
plt.legend()
plt.show()

# Predicting the sine curve with the last window only
current_window = X[-1].reshape(1, 50, 1)

future_predictions = []
n_predictions = 100

for i in range(n_predictions):
    next_value = model_rnn.predict(current_window, verbose=0)
    
    future_predictions.append(next_value[0, 0])
    
    # Updating the window 
    next_value_reshaped = next_value.reshape(1, 1, 1)
    current_window = np.append(current_window[:, 1:, :], next_value_reshaped, axis=1)

# Ploting the results
plt.figure(figsize=(10, 5))
plt.plot(range(50), X[-1], label="Last known window")
plt.plot(range(50, 50 + n_predictions), future_predictions, label="Future Prediction", color='red')
plt.legend()
plt.title("Extrapolation with the RNN")
plt.show()
