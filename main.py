import random
from transformer import DataTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

#random.seed(0)

# maximum number of features to represent each text-snippet with
max_features = 200

# range of n-grams to serach for during training
ngram_range = (1,4)

# number of snippets to extract from each language
data_size = 2000

# the ratio size of the test set
test_size = 0.20



### Learn feature representation of data ###
dt = DataTransformer(max_features)
x_train, y_train, x_test, y_test = dt.split_data(data_size = data_size, test_size=test_size)
dt.fit(x_train, y_train, ngram_range=ngram_range)

# transform input data to their feature representation
print 'transforming data...'
x_train_trans = dt.transform(x_train)
x_test_trans = dt.transform(x_test)

print 'creating and training a gaussian naive bayes classifier...'
gnb = GaussianNB()
gnb.fit(x_train_trans, y_train)


print 'gnb score on test set:', gnb.score(x_test_trans, y_test)

# save classifier for other purposes
joblib.dump(gnb, 'classifier.pkl') 



