from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

'''cross_validation'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .5)

# First method = 0.96
'''
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
'''

# Second method = 0.96
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))