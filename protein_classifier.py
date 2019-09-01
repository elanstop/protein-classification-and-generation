import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import backend as K

file = open('100_to_200_natural.txt','rb')
natural_proteins = pickle.load(file)
file.close()

#random data created by shuffling real protein sequences, so that the amino acid distribution is left fixed (see make_data.py)
#there are two random proteins for each of the natural proteins
file2 = open('100_to_200_random.txt','rb')
random_proteins = pickle.load(file2)
file2.close()

#we truncate the list of random proteins to have the same length as the natural proteins
all_data = list(natural_proteins)+list(random_proteins)[:len(natural_proteins)]

#we previously labeled the data
#pre-labeling can be dropped in a future version
X = np.array([entry[0] for entry in all_data])
Y = np.array([entry[1] for entry in all_data])

x_train, x_val, y_train, y_val = train_test_split(X,Y,test_size = 0.2)

#might be used in the future to benefit from being able to make arbitrary amounts of fake data
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed


model = Sequential()
model.add(LSTM(21,input_shape=(200,21)))
model.add(Dense(2, activation='sigmoid'))


#uncomment line below to begin training on a particularly strong, pre-trained model
#model.load_weights('best_classifier_yet_weights.h5')

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5, batch_size=100,
          validation_data=(x_val,y_val))

model.save('protein_classifier.h5')



