import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

addition = tf.add(a,b)


with tf.Session() as sess:
    sess.run(init)
    print("addition: ",addition,"-",sess.run(addition,feed_dict={a:2,b:3)))

sess.close()