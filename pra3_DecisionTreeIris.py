# https://ai.google/education/#?modal_active=yt-tNa99PG8hR8
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image
import matplotlib.pyplot as plt

iris = load_iris()
test_idx = [0, 25, 50, 75, 100, 125]

# 1. Import dataset
'''
print(iris.feature_names)
print(iris.target_names)

for i in range(len(iris.target)):
    print("Example %d: label %s, feature %s" % (i, iris.target[i], iris.data[i]))
'''

# 2. Train a classifier
# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# 3. Predict label for new flower
print(test_target)
print(clf.predict(test_data))

# 4. Visualize the tree
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True, rounded=True,
                        impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())