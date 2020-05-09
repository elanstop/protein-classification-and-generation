import numpy as np
import pickle
from random import sample, randint
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Lambda, LSTM, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

file = open('new_training_natural_proteins.txt', 'rb')
sequence_data = pad_sequences(pickle.load(file), padding='post', value=0.0)
print(len(sequence_data))
file.close()


# this function is implemented through a Lambda layer. it is needed to ensure that the generator outputs one-hot
# encodings in a differentiable manner. essentially, it is "freezing" a probability distribution onto its
# lowest-energy state, i.e. the entry that initially has the largest value. due to the fact that all distributions
# having a largest entry at the same index will get mapped to the same one-hot vector, this layer might impede the
# gradient, and it will be useful to consider improvements in the future

class GAN:

	def __init__(self, temp=0.001, iterations=2000, batch_size=32, handicap=20, load_weights=True):
		self.temp = temp
		self.iterations = iterations
		self.batch_size = batch_size
		self.handicap = handicap
		self.load_weights = load_weights
		self.G = self.generator()
		self.D = self.discriminator()
		self.stacked_G_D = self.stacked_G_D()

	def one_hot_output(self, x):
		return tf.nn.softmax(x / self.temp)

	@staticmethod
	def random_truncate(set):
		truncated_list = [set[0]]
		for i in range(1, len(set)):
			length = randint(100, 200)
			truncated_list.append(set[i][:length])
		return pad_sequences(truncated_list, padding='post', value=0.0)

	def discriminator(self):
		D = Sequential()
		D.add(Masking(mask_value=0, input_shape=(200, 20)))
		D.add(LSTM(20, return_sequences=True))
		D.add(LSTM(20))
		D.add(Dense(1, activation='sigmoid'))
		if self.load_weights:
			D.load_weights('current_best_classifier.hdf5')
		D.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
		return D

	def generator(self):
		G = Sequential()
		G.add(Reshape((200, 20)))
		G.add(LSTM(20, return_sequences=True))
		G.add(Lambda(self.one_hot_output, trainable=False, name='lambda'))
		G.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
		return G

	def stacked_G_D(self):
		self.D.trainable = False
		model = Sequential()
		model.add(self.G)
		model.add(self.D)
		model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
		return model

	def train(self):
		iterations = self.iterations
		batch_size = self.batch_size
		handicap = self.handicap
		plot_data = []
		positive_labels = np.array([1]*batch_size)
		negative_labels = np.array([0]*batch_size)
		for cnt in range(iterations):

			noise = np.random.normal(0, 1, (batch_size, 4000))
			g_loss = self.stacked_G_D.train_on_batch(noise, positive_labels)

			# we don't update the discriminator as often because it is pre-trained and otherwise dominates the generator
			if cnt % handicap == 0:
				gen_noise = np.random.normal(0, 1, (batch_size // 2, 4000))
				synthetic_proteins = self.random_truncate(self.G.predict(gen_noise))
				natural_proteins = sample(list(sequence_data), batch_size // 2)
				x_combined_batch = np.concatenate((natural_proteins, synthetic_proteins))
				y_combined_batch = np.concatenate(
					(positive_labels[0:batch_size // 2], negative_labels[0:batch_size // 2]))
				d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
				plot_data.append((cnt, g_loss, d_loss))

				print('---------------------')
				print('iteration:', cnt)
				print('generator loss:', g_loss)
				print('discriminator loss:', d_loss)

		return plot_data


gan = GAN()
output_data = gan.train()

file = open('gan_plot_data.txt', 'wb')
pickle.dump(output_data, file)
file.close()

gan.G.save('new_protein_generator.h5')
gan.stacked_G_D.save('new_stacked_G_D.h5')
