from asyncore import read
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab



def tokenize(data):

	bert_tokenizer_params=dict(lower_case=True)
	reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

	bert_vocab_args = dict(
		# The target vocabulary size
		vocab_size = 8000,
		# Reserved tokens that must be included in the vocabulary
		reserved_tokens=reserved_tokens,
		# Arguments for `text.BertTokenizer`
		bert_tokenizer_params=bert_tokenizer_params,
		# Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
		learn_params={},
	)


	pt_vocab = bert_vocab.bert_vocab_from_dataset(
    data.batch(1000).prefetch(2),
    **bert_vocab_args
	)

	print(pt_vocab)
	
	return pt_vocab
	

def build_vocab(sentences):
	"""
	DO NOT CHANGE
  Builds vocab from list of sentences
	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE
  Convert sentences to indexed
	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
	DO NOT CHANGE
  Load text data from file
	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text

def get_data(training_file, testing_file):

	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index

	#TODO:

	train = read_data(training_file)
	tokenize(train)

	'''

	#1) Read English and French Data for training and testing (see read_data)
	train_english = read_data(english_training_file)
	train_french  = read_data(french_training_file)
	test_english  = read_data(english_test_file)
	test_french   = read_data(french_test_file)
	#2) Pad training data (see pad_corpus)
	train_french, train_english  = pad_corpus(train_french, train_english)
	#3) Pad testing data (see pad_corpus)
	test_french, test_english  = pad_corpus(test_french,  test_english)
	#4) Build vocab for French (see build_vocab)
	french_vocab, _ = build_vocab(train_french)
	#5) Build vocab for English (see build_vocab)
	english_vocab, eng_padding_index = build_vocab(train_english)
	#6) Convert training and testing English sentences to list of IDS (see convert_to_id)
	train_english = convert_to_id(english_vocab, train_english)
	test_english  = convert_to_id(english_vocab, test_english)
	#7) Convert training and testing French sentences to list of IDS (see convert_to_id)
	train_french = convert_to_id(french_vocab, train_french)
	test_french  = convert_to_id(french_vocab, test_french)
	return train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index
	'''

data = get_data("../data/friends_transcript.txt", None)
