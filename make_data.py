from Bio import SeqIO
import numpy as np
import pickle
from random import shuffle, seed

#data downloaded in .fasta file format from UniProt

#funky amino letters are X,U,Z,B. We exclude sequences containing these letters.

#100_to_200.fasta was created with the following search terms: length 100 to 200, complete sequences, evidence at protein level, reviewed.
#it was used as the source for the training files 100_to_200_natural.txt and 100_to_200_random.txt

#filename = "100_to_200.fasta"

#100_to_200_transcript_level.fasta was created with the same search terms, only using evidence at transcript level rather than protein level.
#of the 8865 proteins in this file, only 1023 are found in 100_to_200.fasta as well.
#we can therefore use this file for testing
filename = "100_to_200_transcript_level.fasta"


#the maximum length of any protein in the set
#we will pad shorter proteins to reach this length
max_length = 200

#how many random proteins we make for each natural one
multiplicity = 2

#provide one-hot encoding for each of the amino letters
#the final vector is reserved for a padding symbol
def make_amino_dict():
	amino_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
	code_list = []
	for i in range(21):
		code = [0]*21
		code[i] = 1
		code_list.append(code)
	code_dict = {}
	for i in range(20):
		code_dict[amino_list[i]] = code_list[i]
	return code_list,code_dict

code_list, code_dict = make_amino_dict()

def encode():
	sequence_list = []
	bad_letters = set('XUZB')
	for record in SeqIO.parse(filename, "fasta"):
		this_sequence = []
		#exclude sequences containing the funky amino acids
		if any([(c in bad_letters) for c in str(record.seq)]):
			continue
		for letter in record.seq:
			new_letter = code_dict[str(letter)]
			this_sequence.append(new_letter)
		sequence_list.append(this_sequence)
	return sequence_list

sequence_list = encode()

def make_shuffled_sequences():
	shuffled_sequence_list = []
	for sequence in sequence_list:
		for i in range(multiplicity):
				seed()
				new_sequence = sequence.copy()
				shuffle(new_sequence)
				shuffled_sequence_list.append(new_sequence)
	return shuffled_sequence_list

shuffled_sequence_list = make_shuffled_sequences()


def pad_sequences():
	padded_sequences = []
	padded_shuffled_sequences = []
	for sequence in sequence_list:
		while(len(sequence)) < max_length:
			sequence.append(code_list[20])
		padded_sequences.append(np.array(sequence))
	for sequence in shuffled_sequence_list:
		while(len(sequence)) < max_length:
			sequence.append(code_list[20])
		padded_shuffled_sequences.append(np.array(sequence))
	return padded_sequences, padded_shuffled_sequences

padded_sequences, padded_shuffled_sequences = pad_sequences()

#this can be dropped in future version. train_test_split makes the job easy
def label_sequences():
	labeled_sequences = []
	shuffled_labeled_sequences = []
	for sequence in padded_sequences:
		labeled_sequences.append((sequence,[0,1]))
	for sequence in padded_shuffled_sequences:
		shuffled_labeled_sequences.append((sequence,[1,0]))
	return np.array(labeled_sequences), np.array(shuffled_labeled_sequences)

natural_proteins, random_proteins = label_sequences()

# file = open('100_to_200_natural_test_data.txt','wb')
# pickle.dump(natural_proteins,file)
# file.close()

# file2 = open('100_to_200_random_test_data.txt','wb')
# pickle.dump(random_proteins,file2)
# file2.close()









	

