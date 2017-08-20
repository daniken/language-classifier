# -*- coding: utf-8 -*-

import os
import time
import io
import random
import numpy as np

# for all .dat files in ./data/concatenated:
# split the contents of each file into a common training/test set


class DataTransformer():
	
	def __init__(self, max_features=None):	

		self.max_features = max_features			# maximum number of top-terms/features per language to use  
		self.labels = []							# list for mapping labels (int) to language code (str)
		self.n_docs = -1							# number of documents in training corpus
		self.n_total_terms = -1						# total number of terms in training set
		self.n_features = -1						# number of terms/features in final vector

		self.lf = []								# language frequency - in how many languages each term appears
		self.tf = []								# term frequency - how many times each term appear in training data
		self.df = []								# document frequency - amount of times each term appear in training data

		self.term_in_languages = {}					# key = term-index, value = list of language-indices where this term appear

		self.lang_dicts = []						# contains term-count dictionaries for each language
		self.doc_dicts = []							# contains term-count dictionaries for each document

		self.vocabulary	= {}						# key = term, value = unique index, for all terms in train data
		self.features = []							# contains top-terms

	
	# finds the most important terms in @corpus (training set) and saves them in @self.features
	# limits the number of most important terms to @max_features
	def fit(self, corpus, labels, ngram_range=(1,1)):		

		self.generate_terms(corpus, labels, ngram_range)	
		self.compute_frequences()
		self.compute_top_terms()

	
	# go through every language's snippets and:
	# generate ngrams. And for each ngram:
	# count document occurences and language occurences
	def generate_terms(self, corpus, labels, ngram_range):
		print 'generating terms ...'

		# sort data and targets before extracting doc_counts and generating terms
		permutation = np.argsort(labels)
		labels = np.array(labels)[permutation].tolist()
		corpus = np.array(corpus)[permutation].tolist()

		self.labels, doc_counts = np.unique(labels, return_counts=True)
		self.n_docs = sum(doc_counts)
		
		# needs to be iterable if we want loop over @corpus in epochs of every language
		# if data is shuffled we need to sort them by @labels
		docs = iter(corpus)

		for lang_i in self.labels:
			
			lang_dict = {}
			
			for j in range(doc_counts[lang_i]):
				
				doc_dict = {}
				doc = docs.next()
				
				for word in doc.split(' '):	# prevents processing whitespaces
					for dx in range(ngram_range[0], ngram_range[1] + 1):
						for x in range(len(word)-dx+1):
				        
							term = word[x:x+dx]

							# give each term a unique index
							self.vocabulary.setdefault(term, len(self.vocabulary))
		
							# keep track of how many languages this term has appeared in
							l = self.term_in_languages.setdefault(self.vocabulary[term], [])
							if lang_i not in l:
								l.append(lang_i)

							# count occurences of each term for this language/document
							lang_dict[term] = lang_dict.setdefault(term, 0) + 1						
							doc_dict[term] = doc_dict.setdefault(term, 0) + 1
				
					
					self.doc_dicts.append(doc_dict)

			self.lang_dicts.append(lang_dict)
			print 'found %d terms in language %d' % (len(lang_dict), lang_i) 

		
		self.n_total_terms = len(self.vocabulary)
		print 'found %d unique terms in total' % (self.n_total_terms) 
			

	# iterate over all languages and count:
	# the number of times each term appeared in a langauge
	#
	# iterate over all documents and count:
	# the number of times each term appeared in a document
	# the amount of times each term appeared in all documents
	def compute_frequences(self):
		print 'computing frequencies...'

		self.lf = [0] * self.n_total_terms			#	lf =  language frequency
		self.tf = [0] * self.n_total_terms			#	tf  = term frequency	
		self.df = [0] * self.n_total_terms			#	df  = document frequency
	
		for term, index in self.vocabulary.iteritems():
			self.lf[index] = len(self.term_in_languages[index])
		

		for doc_dict in self.doc_dicts:
			for term, count in doc_dict.iteritems():
				self.df[self.vocabulary[term]] += 1
				self.tf[self.vocabulary[term]] += count


	def compute_top_terms(self):
		print 'computing top terms...'

		### Compute new ranking ###
		top_term_count_by_lang, next_top_term_count_by_lang = self.get_tops_occurences()


		terms = [0] * self.n_total_terms
		rank = [0] * self.n_total_terms

		for term, index in self.vocabulary.iteritems():
			terms[index] = term


			# precomputations
			tl = len(term)
			lang_uniqueness = float(top_term_count_by_lang[index]) / max(0.5, float(next_top_term_count_by_lang[index]))
			tf = float(self.tf[index])
			lf = float(self.lf[index])
			df = float(self.df[index])
			idf = np.log(tl * float(self.n_docs) / df)

			rank[index] = lang_uniqueness * np.log(tf) * (tl/lf)**2

		# sort by new ranking
		rank, terms = (list(t) for t in zip(*sorted(zip(rank, terms), reverse=True)))

		## add language unique unigrams to the top features before clipping it###
		for i in range(self.n_total_terms):
			if self.lf[self.vocabulary[terms[i]]] == 1 and len(terms[i]) == 1 and self.tf[self.vocabulary[terms[i]]] > 1:
				term = terms[i]
				del terms[i] 							
				terms = [term] + terms

		### extract top @self.max_features from each language
		#top_features = []
		#for lang_i in self.labels:
		#
		#	print '\ntop %d features from language %d: ' % (self.max_features, lang_i)
		#	counter = 0
		#	for i in range(len(terms)):
		#
		#		if lang_i in self.term_in_languages[self.vocabulary[terms[i]]] and len(self.term_in_languages[self.vocabulary[terms[i]]]) == 1:
		#			top_features.append(terms[i])
		#
		#			print '%d. %s' % (i, terms[i])
		#
		#			counter += 1
		#			if counter >= self.max_features:
		#				break
					

		## handle final number of top-terms ###
		if self.max_features == None:
			self.max_features = len(terms)
		else:
			if self.max_features > self.n_total_terms:
				self.max_features = self.n_total_terms

		self.n_features = self.max_features
		#self.n_features = len(top_features)
 
		### append and save final features ###
		for i in range(self.n_features):
			self.features.append(terms[i])

		#for term in top_features:
		#	self.features.append(term)

		self.save_features()

		# only for printing
		print 'top %d features to be used to train MLP:' % (self.n_features)
		for i in range(len(self.features)):
		
			languages = ''
			for l in self.term_in_languages[self.vocabulary[terms[i]]]:
				languages += str(l) + ' '

			print 'rank: %4d term: %s\t lf: %d\t df: %4d\t tf: %3d\t l:%s' % (
i+1,
terms[i].encode('utf-8'),
self.lf[self.vocabulary[terms[i]]],
self.df[self.vocabulary[terms[i]]],
self.tf[self.vocabulary[terms[i]]],
languages)


		
		
	def save_features(self):

		with io.open('features/features.txt', 'w+', encoding='utf-8') as feature_file:
			for feature in self.features:
				feature_file.write(feature + '\n')


	def load_features(self):
		
		with io.open('features/features.txt', 'r', encoding='utf-8') as feature_file:
			for line in feature_file:
				self.features.append(line.rstrip('\n'))	# don't strip away whitespace that is part of the feature

		self.n_features = len(self.features)



	# returns two lists:
	# @top_counts		- the amount of times each term appear in the most frequent language
	# @next_top_counts	- the amount of times each term appear in the next to most frequent language
	def get_tops_occurences(self):

		top_counts = [-1] * self.n_total_terms
		top_langs = [-1] * self.n_total_terms

		next_top_counts = [-99] * self.n_total_terms
		next_top_langs = [-1] * self.n_total_terms

		for lang_i in self.labels:
			for term, count in self.lang_dicts[lang_i].iteritems():
					
				index = self.vocabulary[term]

				if count > top_counts[index] or top_langs[index] == -1:

				
					next_top_counts[index] = top_counts[index]
					next_top_langs[index] = top_langs[index]

					top_counts[index] = count
					top_langs[index] = lang_i

				else:
					if count >= next_top_counts[index]:
						next_top_counts[index] = count
						next_top_langs[index] = lang_i
			
		for i in range(self.n_total_terms):
			if next_top_counts[i] == -1:
				next_top_counts[i] = 0
			
		return top_counts, next_top_counts



	# creates and returns a document-term matrix of size (n_documents x n_features) out of @corpus
	# each element corresponds to the amount of times the term of that column appear in the document of that row
	def transform(self, corpus):
	
		
		feature_matrix = [0] * len(corpus)
		
		# go through every document
		for i in range(len(corpus)):
		
			feature_matrix[i] = [0] * self.n_features
			doc = corpus[i].lower()

			# for every top feature
			for j in range(self.n_features):

				feature_matrix[i][j] = get_termcount(doc, self.features[j])


		return feature_matrix
						

	def split_data(self, data_size=None, test_size=0.5):
	
		
		# extract equal amount of lines from each langauge
		least_lines = min([get_linecount('data/concatenated/'+f) for f in os.listdir('data/concatenated/')])
		if data_size == None:
			max_lines = least_lines
		else:
			max_lines = min(data_size, least_lines)
		print 'maximum number of lines to extract in each file:', max_lines

		x_train, y_train = [], []
		x_test, y_test = [], []

		lang_counter = 0		# keeps track of label id

		for file_name in os.listdir('data/concatenated/'):

			if file_name.split('.')[-1] == 'dat':
		
				# computes training/test sizes for this particular language
				n_lines = max_lines
				n_test = int(n_lines * test_size)
				n_train = n_lines - n_test

				with io.open('data/concatenated/' + file_name, 'r', encoding='utf-8') as data:
			

					# add first @n_train lines to training set
					for i in range(n_train):
					
						word = data.next().strip()
						x_train.append(word)
						y_train.append(lang_counter)

	
					# add remaining @n_test lines to test set
					for i in range(n_test):
					
						word = data.next().strip()
						x_test.append(word)
						y_test.append(lang_counter)	

					lang_counter += 1

				print 'language %s with %d sentences split into n_training=%d, n_test=%d' % (
file_name.split('.')[0],
n_test + n_train,
n_train,
n_test)

		# only for printing
		n_train_samples = len(x_train)
		n_test_samples = len(x_test)
		n_samples = n_train_samples + n_test_samples
		print 'total number of samples:', n_samples
		print 'total number of training samples:', n_train_samples
		print 'total number of test samples: ', n_test_samples
		return x_train, y_train, x_test, y_test

	
def get_termcount(doc, term_in_quest):
	term_length = len(term_in_quest)
	count = 0

	try:
		doc = doc.decode('utf8')
	except:
		pass

	for x in range(len(doc)-term_length+1):
		term = doc[x:x+term_length]

		if term == term_in_quest:
			count +=1

	return count
	

# computes and returns the number of lines in a file using buffering techniques
def get_linecount(file_path):
	f = open(file_path)                  
	lines = 0
	buf_size = 1024 * 1024
	read_f = f.read

	buf = read_f(buf_size)
	while buf:
		lines += buf.count('\n')
		buf = read_f(buf_size)
	f.close()
	return lines


## range of n-grams to serach for during training
ngram_range = (1,4)

## number of snippets to extract from each language
data_size = 2000

## the ratio size of the test set
test_size = 0.60

## maximum number of features to represent each text-snippet with
max_features = 20



if __name__ == '__main__':

	### fit data to transformer only ###
	#transformer = DataTransformer(max_features)
	#x_train, y_train, x_test, y_test = transformer.split_data(data_size=data_size, test_size=test_size)
	#transformer.fit(x_train, y_train, ngram_range=ngram_range)
	pass







