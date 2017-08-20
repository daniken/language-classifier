# -*- coding: utf-8 -*-
import io
import re
import os
import random


# symbol regex pattern
symbol_reg = re.compile('[{}()\[\]@.,/*?´`\-+$!%#^&;\\|<£«>"\'º_\d=:§]')


#    language specific regex objects
#
#    browse to https://www.wikipedia.org/ and search on your language's orthography 
#    browse to https://unicode-table.com/en/#latin-1-supplement and match all characters with corresponding unicode

sv_reg = re.compile(ur'[^\u0061-\u007A\u00E5\u00E4\u00F6\u00C5\u00C4\u00D6]')
de_reg = re.compile(ur'[^\u0061-\u007A\u00C4\u00E4\u00D6\u00F6\u00DC\u00FC\u1E9E\u00DF]')
it_reg = re.compile(ur'[^\u0061-\u0076\u00E0\u00E8\u00E9\u00EC\u00ED\u00F2\u00F3\u00F9\u00FA]')
fr_reg = re.compile(ur'[^\u0061-\u007A\u00E0\u00E2\u00E6\u00E7\u00E8\u00E9\u00EA\u00EB\u00EE\u00EF\u00F4\u0153\u00F9\u00FB\u00FC\u00FF]')
#xx_reg = re.compile(ur'[^\u]')	# template


# dictionary for picking correct language regex pattern
lang_regs = {'SV' : sv_reg, 'DE' : de_reg, 'IT' : it_reg, 'FR' : fr_reg}




# computes and returns the number of lines in a file using buffering techniques
def get_linecount(file_path):
    with open(file_path) as f:  
         
		n_lines = 0
		buf_size = 1024 * 1024
		read_f = f.read

		buf = read_f(buf_size)
		while buf:
		    n_lines += buf.count('\n')
		    buf = read_f(buf_size)

    return lines





### remove words with bad/illegal characters ###
def clean_data():

	for file_name in os.listdir('data/original'):
		if file_name.split('.')[-1] == 'dat':

			lang_code = file_name.split('.')[0]

			chars = set()
			words = set()

			counter = 0

			max_words = min(max_words, get_linecount('data/original/' + file_name))

			with io.open('data/original/' + file_name, 'r', encoding='utf-8') as original_file:
				with io.open('data/clean/' + file_name, 'w+', encoding='utf-8') as clean_file:
					for line in original_file:

						word = line.strip().lower()

						# filter out words containing symbols
						if symbol_reg.search(word) is None:

							# filter out words with letters outside of the langauge
							if lang_code in lang_regs:
								if lang_regs[lang_code].search(word) is None:

									# filter out words that have already appeared
									# faster on average to filter this way 
									if word in words:
										continue
									else:
										clean_file.write(word + '\n')
										words.add(word)

										counter += 1

										if counter >= max_words:
											break

					# append word set for cross uniquifying later
					words_list.append(words)

					print 'found %d unique words in language %s' % (len(words), lang_code)



### remove words that overlap between languages ###
def cross_uniquify_data():

	languages = [file_name.split('.')[0] for file_name in os.listdir('data/clean') if file_name.split('.')[-1] == 'dat']

	for i in range(len(languages)):

		words_i = words_list[i]

		for j in range(i+1, len(languages)):    # don't compare to already compared languages

			words_j = words_list[j]
			duplicates = words_i.intersection(words_j)

			print 'number of intersecting words removed from language %s/%s: %d' % (languages[i],languages[j],len(duplicates))

			# remove from both languages
			for k in [i,j]:
				with io.open('data/clean/' + languages[k] + '.dat', 'r', encoding='utf-8') as original_file:
					with io.open('data/cross-unique/' + languages[k] + '.dat', 'w+', encoding='utf-8') as new_file:
						for line in original_file:

							if line.strip('\n') in duplicates:
								continue
							else:
								new_file.write(line)

    
    

### concatenate words into snippets ###
def concatenate_data():
	for file_name in os.listdir('data/cross-unique'):
		if file_name.split('.')[-1] == 'dat':

			with io.open('data/cross-unique/' + file_name, 'r', encoding='utf-8') as original_file:

				concatenated_words = []
				lines = get_linecount('data/cross-unique/' + file_name)
				lang_code = file_name.split('.')[0]

				### as @n_words being random, subtract processed lines until we reach e.o.f. ###
				while lines >= min_words:

					n_words = random.randint(min_words, max_words)
					doc = ''

					# make sure we don't iterate more than we should
					if lines < n_words:
						n_words = lines

					for i in range(n_words):
						word = original_file.next().strip()
						doc += word + ' '

					concatenated_words.append(doc.strip())
					lines = lines - n_words

			### add all snippets to new file ###
			with io.open('data/concatenated/' + file_name, 'w+', encoding='utf-8') as new_file:
				for snippet in concatenated_words:
					new_file.write(snippet + '\n')

			print 'concatenated words into %d snippets in language %s' % (len(concatenated_words), lang_code)



# maximum number of raw words to process
max_words = 1000

# list containing words sets of all languages
words_list = []

# minimum/maximum number of words per snippet
min_words = 10
max_words = 20


if __name__ == '__main__':
	
	### preprocess only ###
	#clean_data()
	#cross_uniquify_data()
	#concatenate_data()
	pass



