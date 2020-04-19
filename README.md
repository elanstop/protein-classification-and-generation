# protein-classification-and-generation
Using deep learning to distinguish naturally occurring protein sequences from randomly shuffled ones

# Motivation and Results
The extent to which protein amino acid sequences found in nature differ from random sequences reamins an open question. Using an LSTM-based neural network, I have shown that sequences can be distinguished with record ~98% accuracy. This is significantly higher than previous benchmarks I'm aware of, and relies solely on the information content of the sequences themselves without reference to chemical properties. See, e.g., [here](https://www.sciencedirect.com/science/article/abs/pii/S0022519315005731), [here](https://www.biorxiv.org/content/10.1101/687558v2.abstract), and [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3353917/) for related work. Obtaining a NN model that can reliably distinguish natural proteins from random ones lays the foundation for a GAN that can propose new proteins not found in nature (coming soon).

# Use
Run make_data.py to preprocess the raw data contained in the included .fasta files, which were downloaded from UniProt. Query information used to make these files can be found in make_data.py

Run protein_classifier.py to train on the processed data. Note that training will be much faster with a GPU. Saved models can be tested by running testing.py

(protein_GAN.py is work in progress)
