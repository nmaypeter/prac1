# https://www.youtube.com/watch?v=cSKfRcEDGUs&index=6&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
#print(hello)