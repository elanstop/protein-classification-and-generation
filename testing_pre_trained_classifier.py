import tensorflow as tf
import pickle
from random import shuffle
import numpy as np

#accuracy on this test set is 0.9769
#however, 1023 of the 8865 proteins in this set might have also been present during training
#so true accuracy likely a bit lower

file1 = open('/Users/elanstopnitzky/desktop/science_projects/protein_classification/data/100_to_200_natural_test_data.txt','rb')
natural_proteins = pickle.load(file1)
file1.close()


file2 = open('/Users/elanstopnitzky/desktop/science_projects/protein_classification/data/100_to_200_random_test_data.txt','rb')
random_proteins = pickle.load(file2)
file2.close()

all_data = list(natural_proteins)+list(random_proteins)[:len(natural_proteins)]

data = np.array([x[0] for x in all_data])
labels = np.array([x[1] for x in all_data])

model = tf.keras.models.load_model('best_classifier_yet.h5')

model.evaluate(data,labels)



