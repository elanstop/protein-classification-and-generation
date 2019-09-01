import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Reshape, LSTM, Lambda
from tensorflow.python.keras.optimizers import Adam
import pickle

#examine the output of the generator
#note: the generator does not currently pad correctly (the pad symbol is denoted by the digit 21 in the output)
#a question mark denotes an output that could not be interpreted as a one-hot encoding of one of the amino acids (rare at low temp)

file = open('100_to_200_natural.txt','rb')
data = pickle.load(file)
sequence_data = [x[0] for x in data]
file.close()

#must use same value of temp as that present during training
temp = 0.001
def one_hot_output(x):
	return tf.nn.softmax(x/temp)


code_list = []
for i in range(21):
	code = [0]*21
	code[i] = float(1)
	code_list.append(code)

def generator_output(batch):
	saved_generator = tf.keras.models.load_model('protein_generator.h5')
	noise = np.random.normal(0, 1, (batch,100))
	x = tf.constant(noise)
	output = saved_generator(x)
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	q = sess.run(output)
	for i in range(batch):
		print('~~~~~~~~~~~~~~~~~~~~~~~~~')
		seq = ''
		for j in range(200):
			letter = list(q[i][j])
			rounded_list = [np.round(x) for x in letter]
			multiplicity = len([x for x in rounded_list if x==1])
			if multiplicity == 1:
				identity = rounded_list.index(1)
				seq += str(identity)+'-'
			else:
				seq += '?'
		print(seq)





generator_output(100)




