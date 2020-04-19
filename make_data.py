from Bio import SeqIO
import numpy as np
import pickle
from random import shuffle, seed


# data downloaded in .fasta file format from UniProt

# funky amino letters are X,U,Z,B. We exclude sequences containing these letters.

# 100_to_200.fasta was created with the following search terms: length 100 to 200, complete sequences, evidence at
# protein level, reviewed. it was used as the source for the training files 100_to_200_natural.txt and
# 100_to_200_random.txt

# 100_to_200_transcript_level.fasta was created with the same search terms, only using evidence at transcript level
# rather than protein level. of the 8865 proteins in this file, only 1023 are found in 100_to_200.fasta as well. we
# can therefore use this file for testing

class Preprocess:
    def __init__(self, data_type, natural_output_file, random_output_file, raw_train_data='100_to_200.fasta',
                 raw_test_data="100_to_200_transcript_level.fasta"):
        self.data_type = data_type
        self.natural_output_file = natural_output_file
        self.random_output_file = random_output_file
        self.raw_train_data = raw_train_data
        self.raw_test_data = raw_test_data
        self.code_dict = self.make_amino_dict()
        self.input_sequences = self.extract_sequences()
        self.encoded_sequences = self.encode()
        self.encoded_shuffled_sequences = self.shuffle_sequences()

    @staticmethod
    def make_amino_dict():
        amino_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y', '0']
        one_hots = np.eye(21, 21)
        code_list = [one_hots[i] for i in range(20)]
        code_dict = dict(zip(amino_list, code_list))
        return code_dict

    def extract_sequences(self):
        sequence_list = []
        bad_letters = set('XUZB')
        if self.data_type == 'training':
            data = self.raw_train_data
        if self.data_type == 'testing':
            data = self.raw_test_data
        for record in SeqIO.parse(data, "fasta"):
            # exclude sequences containing the funky amino acids
            if any([(c in bad_letters) for c in str(record.seq)]):
                continue
            sequence_list.append(list(str(record.seq)))
        return sequence_list

    def encode(self):
        encoded_sequence_list = []
        for sequence in self.input_sequences:
            this_sequence = []
            for letter in sequence:
                new_letter = self.code_dict[letter]
                this_sequence.append(new_letter)
            encoded_sequence_list.append(this_sequence)
        return encoded_sequence_list

    def shuffle_sequences(self):
        shuffled_sequence_list = []
        for sequence in self.encoded_sequences:
            seed()
            new_sequence = sequence.copy()
            shuffle(new_sequence)
            shuffled_sequence_list.append(new_sequence)
        return shuffled_sequence_list

    def save(self):
        file = open(self.natural_output_file, 'wb')
        pickle.dump(self.encoded_sequences, file)
        file.close()

        file2 = open(self.random_output_file, 'wb')
        pickle.dump(self.encoded_shuffled_sequences, file2)
        file2.close()


training_data = Preprocess('training', 'new_training_natural_proteins.txt', 'new_training_random_proteins.txt')
training_data.save()

testing_data = Preprocess('testing', 'new_testing_natural_proteins.txt', 'new_testing_random_proteins.txt')
testing_data.save()
