# language-classifier
Naive Bayes classifier for identifying language of text snippets


* With `preprocess.py` you can clean data from words containing symbols, filter words by the corresponding language's alphabet, remove cross-duplicate words and concatemate words into snippets of specified length.

* With `transformer.py` you can split data into specified training/test set, learn a feature representation from training data and be able to transform snippets into this learned representation.

* With `main.py` you can either perform what `transformer.py` does or load already leared features and train a naive bayes classifier.



















