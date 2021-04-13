import numpy as np
import tensorflow as tf

npz = np.load('Audiobook_data_train.npz')

train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

npz = np.load('Audiobook_data_validation.npz')

validation_inputs = npz['inputs'].astype(np.float)
validation_targets = npz['targets'].astype(np.int)

npz = np.load('Audiobook_data_test.npz')

test_inputs = npz['inputs'].astype(np.float)
test_targets = npz['targets'].astype(np.int)

input_size = 10
output_size = 2
hidden_layer_size = 50

model = tf.keras.Sequential([
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

BATCH_SIZE = 100

MAX_NUM_EPOCHS = 100

model.fit(train_inputs,
          train_targets,
          batch_size=BATCH_SIZE,
          epochs=MAX_NUM_EPOCHS,
          validation_data=(validation_inputs, validation_targets),
          verbose=2)

