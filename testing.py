import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# accuracy on this test set is 98%
# however, 1023 of the 8795 proteins in this set might have also been present during training
# so true accuracy likely slightly lower

file1 = open('new_testing_natural_proteins.txt', 'rb')
natural_proteins = pickle.load(file1)
file1.close()

file2 = open('new_testing_random_proteins.txt', 'rb')
random_proteins = pickle.load(file2)
file2.close()

data = np.array(natural_proteins+random_proteins)
data = pad_sequences(data, padding='post', value=0.0)
labels = np.array([1] * len(natural_proteins) + [0] * len(random_proteins))

model = tf.keras.models.load_model('new_model.100-0.97.hdf5')
model.evaluate(data, labels)