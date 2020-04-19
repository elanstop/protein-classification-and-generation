import numpy as np
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class ProteinClassifier:
    def __init__(self, natural_input='new_training_natural_proteins.txt',
                 random_input='new_training_random_proteins.txt', num_data=15000, test_fraction=0.2, batch_size=100,
                 epochs=20):
        self.natural_input = natural_input
        self.random_input = random_input
        self.num_data = num_data
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.epochs = epochs
        self.natural_proteins, self.random_proteins = self.load_data()
        self.x_train, self.x_test, self.y_train, self.y_test = self.split()
        self.model = self.model()
        self.train = self.train()

    def load_data(self):
        file = open(self.natural_input, 'rb')
        natural_proteins = pickle.load(file)
        print(len(natural_proteins))
        print(type(natural_proteins))
        file.close()

        file2 = open(self.random_input, 'rb')
        random_proteins = pickle.load(file2)
        print(len(random_proteins))
        print(type(random_proteins))
        file2.close()

        return natural_proteins, random_proteins

    def split(self):
        size = int(self.num_data / 2)
        x = np.array(self.natural_proteins[:size] + self.random_proteins[:size])
        print(len(x))
        y = np.array([1] * size + [0] * size)
        print(len(y))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_fraction)

        return pad_sequences(x_train, padding='post', value=0.0), pad_sequences(x_test, padding='post'), y_train, y_test

    @staticmethod
    def model():
        model = Sequential()
        model.add(Masking(mask_value=0, input_shape=(200, 21)))
        model.add(LSTM(20, return_sequences=True))
        model.add(LSTM(20, return_sequences=True))
        model.add(LSTM(20))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        model_checkpoint = ModelCheckpoint('new_model.{epoch:02d}-{val_acc:.2f}.hdf5')
        model = self.model
        x_train, y_train = self.x_train, self.y_train
        x_test, y_test = self.x_test, self.y_test
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_test, y_test),
                  callbacks=[model_checkpoint])


ProteinClassifier()
