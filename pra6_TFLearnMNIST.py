# https://www.youtube.com/watch?v=Gj0iyo265bc&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=7
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

# Import the dataset
mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asanyarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asanyarray(mnist.test.labels, dtype=np.int32)

max_examples = 10000
data = data[:max_examples]
labels = labels[:max_examples]

'''
# Visualize images
def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray_r)

display(0)
print(len(data[0]))
'''

# Initialize the classifier
feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(n_classes=10, feature_columns=feature_columns)
classifier.fit(data, labels, batch_size=100, steps=1000)

# Evaluate accuracy
classifier.evaluate(test_data, test_labels)
'''print(classifier.evaluate(test_data, test_labels)["accuracy"])'''

# Classify the examples
''' # no work
print("Predicted %d, Label: %d" % (classifier.predict(test_data[0]), test_labels[0]))
display(0)
'''
'''
# Visualize learned weights
weights = classifier.weights_
f, axes = plt.subplot(2, 5, figsize=(10, 4))
for i in range(len(axes)):
    a = axes[i]
    a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(())
    a.set_yticks(())
plt.show()
'''