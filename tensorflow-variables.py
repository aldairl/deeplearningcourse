import tensorflow as tf

constant = tf.constant( [5, 3.0, 4.5], dtype= tf.float32, name= 'Constant_init')

#print(constant)

placeholder = tf.placeholder(dtype= tf.float32, name= "Placeholder_init")
#print(placeholder)

#contenido = 3
#variable = tf.Variable(3, dtype= tf.float32, name= 'variable_init')
variable = tf.Variable(3, dtype = tf.float32, name= 'variable_init')
#print(variable)

matrix = tf.zeros([3,4], tf.int32, name= 'matriz_zeros')
#print(matrix)



mult = placeholder*constant
#result = session.run(mult, feed_dict= {placeholder:[[15, 1, 10]]})
#print(result)

A = tf.placeholder(tf.float32, shape=(2,2))
B = tf.placeholder(tf.float32, shape=(2,3))

'''
[1,0]
[0,0]

[1,0]
[0,0]
[1,7]
'''

#dot = tf.tensordot
#producto cruz
mulc = tf.matmul(A, B)


init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(mulc, feed_dict={A:[[1,2],[2,3]], B:[[1,2,3],[4,5,6]]}))


C = tf.placeholder(tf.float32, shape=(2))
D = tf.placeholder(tf.float32, shape=(2))

dot = tf.tensordot(C, D, 1)

session = tf.Session()
session.run(init)
print(session.run(dot, feed_dict= {C:[1,2], D:[4,5]}))