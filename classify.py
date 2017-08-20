# -*- coding: utf-8 -*-

from transformer import DataTransformer
import sys
import os
import io
from sklearn.externals import joblib


# classify all snippets in the .dat file provided via terminal 
# if two .dat files are provided:
# 	first file should contain snippets to classify
#	second file should contain language codes (answers)
#	example:
#
#   $	python classify.py ~/path_to_snippets/test.dat ~/path_to_answers/answers.dat
#
# if a snippet/snippets were provided. it classifies them one by one
#	example:
#
# 	$	python classify.py "this is a sentence" "this is another sentence"
#


if __name__ == "__main__":
	
	### create the mapping between label (int) and language codes (str) ###
	label_to_lang = []
	for file_name in os.listdir('data/concatenated'):
		if file_name.split('.')[-1] == 'dat':
			label_to_lang.append(file_name.split('.')[0])
	
	### process snippets to classifiy depending on how they were provided ###
	if sys.argv[1][-4:] == '.dat':

		### read text snippets from provided .dat file ###
		x = []
		with io.open(sys.argv[1], 'r', encoding='utf-8') as input_file:	
			for line in input_file:
				x.append(line.strip('\n'))

	else:
		x = sys.argv[1:]
	
	### transform snippets to its vector representation learnd from training data ###
	transformer = DataTransformer()
	transformer.load_features()
	x_trans = transformer.transform(x)

	### load classifier and classify snippets ###
	clf = joblib.load('classifier.pkl') 
	preds = clf.predict(x_trans)
	probs = clf.predict_proba(x_trans)


	### print accuracy if file with anwers were provided ###
	if len(sys.argv[1:]) == 2 and sys.argv[2][-4:] == '.dat':

		# convert language code (str) to label (int)
		y = []
		with io.open(sys.argv[2], 'r', encoding='utf-8') as answers_file:
			for line in answers_file:
				y.append(label_to_lang.index(line.strip('\n')))

		for i in range(len(preds)):
		
			ans = '-'
			if y[i] == preds[i]:
				ans = '+'
			print '%1s Lang %2s with %3d %% certainty. %s' % (ans, label_to_lang[preds[i]],  int(100*max(probs[i])), x[i])

		print 'clf score:', clf.score(x_trans, y)

	else:
		for i in range(len(preds)):
			print 'Lang %2s with %.2f %% certainty. %s' % (label_to_lang[preds[i]],  max(probs[i]), x[i])

