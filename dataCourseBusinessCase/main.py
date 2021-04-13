# We want to build a ML algorithm based on our data that can predict
# whether a customer is likely to buy again from the audiobook company.
# The main idea is to prevent the audiobook company spending its advertising budget
# targeting individuals that are unlikely to come back.
# If we can focus our efforts on customers likely to convert again
# we can obtain improved sales and profitability figures.
# Our model will take several metrics and try to predict human behaviour.

# TASK: CREATE A MACHINE LEARNING ALGORITHM THAT CAN PREDICT IF A CUSTOMER WILL BUY AGAIN.

# THE BUSINESS CASE ACTION PLAN:
#
# 1. PREPROCESS THE DATA
# 1.1. balance the dataset
# 1.2. divide the dataset into training, validation and test
# 1.3. save the data in a tensor-friendly format
#
# 2. CREATE MACHINE LEARNING ALGORITHM

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing

raw_csv_data = np.loadtxt(r'C:\Users\User\Downloads\Audiobooks_data.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:, 1:-1]
targets_all = raw_csv_data[:, -1]

num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[: train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

print(np.sum(train_targets), train_samples_count, np.sum(train_targets)/train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets)/validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets)/test_samples_count)

np.savez('Audiobook_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobook_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobook_data_test', inputs=test_inputs, targets=test_targets)

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

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(train_inputs,
          train_targets,
          batch_size=BATCH_SIZE,
          epochs=MAX_NUM_EPOCHS,
          callbacks=[early_stopping],
          validation_data=(validation_inputs, validation_targets),
          verbose=2)

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))







