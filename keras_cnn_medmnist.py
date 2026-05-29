import tensorflow as tf
from tensorflow import keras
import medmnist
from medmnist import INFO
from tensorflow.keras.callbacks import EarlyStopping

data_flag = 'pneumoniamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', download=True)
val_dataset = DataClass(split='val', download=True)
test_dataset = DataClass(split='test', download=True)

x_train, y_train = train_dataset.imgs / 255.0, train_dataset.labels
x_val, y_val = val_dataset.imgs / 255.0, val_dataset.labels
x_test, y_test = test_dataset.imgs / 255.0, test_dataset.labels

ini = keras.layers.Input(shape = x_train.shape[1:])

flat1 = keras.layers.Flatten()(ini)
hidden1 = keras.layers.Dense(128, activation = 'relu')(flat1)
drop1 = keras.layers.Dropout(0.5)(hidden1)

c = keras.layers.Reshape((28, 28, 1))(ini) 
c = keras.layers.Conv2D(32, (3, 3), activation='relu')(c)
c = keras.layers.MaxPooling2D((2, 2))(c)
c = keras.layers.Flatten()(c) # Achata só o resultado dos filtros
c = keras.layers.Dense(64, activation='relu')(c)
c = keras.layers.Dropout(0.5)(c)
concat = keras.layers.concatenate([drop1, c])
combined = keras.layers.Dense(32, activation = 'relu')(concat)
drop = keras.layers.Dropout(0.5)(combined) 
output = keras.layers.Dense(1, activation = "sigmoid")(drop)
model = keras.models.Model(inputs = [ini], outputs=[output])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',    
    patience=5,            
    restore_best_weights=True
)

weights = {0: 3.0, 1:1.0}

history = model.fit(
    x_train, y_train,
    epochs=50,             
    validation_data=(x_val, y_val),
    callbacks=[early_stop], 
    class_weight = wheights
)
