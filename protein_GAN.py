import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Reshape, Lambda, LSTM, BatchNormalization
from tensorflow.python.keras.optimizers import Adam
import pickle

file = open('100_to_200_natural.txt', 'rb')
data = pickle.load(file)
sequence_data = [x[0] for x in data]
print(len(sequence_data))
sequence_labels = [x[1] for x in data]
file.close()


# this function is implemented through a Lambda layer. it is needed to ensure that the generator outputs one-hot
# encodings in a differentiable manner. essentially, it is "freezing" a probability distribution onto its
# lowest-energy state, i.e. the entry that initially has the largest value. due to the fact that all distributions
# having a largest entry at the same index will get mapped to the same one-hot vector, this layer might impede the
# gradient, and it will be useful to consider improvements in the future


def one_hot_output(x, temp=0.001):
	return tf.nn.softmax(x / temp)


class GAN:

	def __init__(self):
		self.G = self.generator()
		self.D = self.discriminator()
		self.stacked_G_D = self.stacked_G_D()

	@staticmethod
	def discriminator():
		D = Sequential()
		D.add(LSTM(21, input_shape=(200, 21)))
		D.add(Dense(2, activation='sigmoid'))
		# load weights from the best discriminator we've yet found
		D.load_weights('best_classifier_yet_weights.h5')
		D.compile(loss='binary_crossentropy', optimizer=Adam())
		return D

	@staticmethod
	def generator():
		G = Sequential()
		G.add(Dense(100, input_shape=(100,), activation='relu', name='dense_1'))
		G.add(BatchNormalization())
		G.add(Dense(100, activation='relu', name='dense_2'))
		G.add(BatchNormalization())
		G.add(Dense(4200, activation='relu', name='dense_3'))
		G.add(Reshape((200, 21), name='reshape'))
		G.add(Lambda(one_hot_output, trainable=False, name='lambda'))
		G.compile(loss='binary_crossentropy', optimizer=Adam())
		return G

	def stacked_G_D(self):
		self.D.trainable = False
		model = Sequential()
		model.add(self.G)
		model.add(self.D)
		model.compile(loss='binary_crossentropy', optimizer=Adam())
		return model

	def train(self, iterations, batch_size, handicap):
		plot_data = []
		positive_labels = np.array([np.array([0, 1]) for _ in range(batch_size)])
		negative_labels = np.array([np.array([1, 0]) for _ in range(batch_size)])
		for cnt in range(iterations):

			noise = np.random.normal(0, 1, (batch_size, 100))
			g_loss = self.stacked_G_D.train_on_batch(noise, positive_labels)

			# we don't update the discriminator as often because it is pre-trained and otherwise dominates the generator
			if cnt % handicap == 0:
				gen_noise = np.random.normal(0, 1, (batch_size // 2, 100))
				synthetic_proteins = self.G.predict(gen_noise)
				# we pull out a random slice of the natural proteins for training, but do not shuffle
				# this can be improved in the future
				random_index = np.random.randint(0, len(sequence_data) - batch_size // 2)
				natural_proteins = sequence_data[random_index:random_index + batch_size // 2]
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

output_data = gan.train(200, 100, 20)

file = open('gan_plot_data.txt', 'wb')
pickle.dump(output_data, file)
file.close()

gan.G.save('new_protein_generator.h5')
gan.stacked_G_D.save('new_stacked_G_D.h5')
