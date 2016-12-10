import scipy as sp
import naive_bayes as nb


data = sp.genfromtxt("nb_training_data.txt", delimiter="\t")
#data features(0 - no, 1 ,yes):
	#chills
	#runny nose
	#headache
	#fever
#y - have flu or not

X = data[:, :-1]
y = data[:, -1]

train_size = int(0.8 * X.shape[0])

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

clf = nb.NaiveBayes()
clf.fit(X, y)

print sum(clf.predict(X_test) == y_test) / (float(X_test.shape[0]))
