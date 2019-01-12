
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf

# Parameters
learning_rate=0.1
epoches=1500
minibatch_size=128
display_epoch=100

# Network Parameters
layer_dims=[784,1024,10]

# tf Graph input
X = tf.placeholder("float", [None, layer_dims[0]])
Y = tf.placeholder("float", [None, layer_dims[len(layer_dims)-1]])

# Store layers weight & bias
parameters={}
L=len(layer_dims)
for l in range(1,L):
    parameters['W'+str(l)]=tf.Variable(tf.random_normal([layer_dims[l-1],layer_dims[l]]))
    parameters['b'+str(l)]=tf.Variable(tf.random_normal([layer_dims[l]]))

# Create model
def neural_net():
    L0=len(parameters)//2
    A=X
    for l in range(1,L0):
        A_prev=A
        Z=tf.add(tf.matmul(A_prev,parameters['W'+str(l)]),parameters['b'+str(l)])
        A=tf.nn.relu(Z)
    Z=tf.add(tf.matmul(A,parameters['W'+str(L0)]),parameters['b'+str(L0)])
    return Z

# Construct model
AL=neural_net()

# Define loss and optimizer
loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=AL,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred=tf.equal(tf.argmax(AL,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for epoch in range(epoches):
        minibatch_X, minibatch_Y=mnist.train.next_batch(minibatch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X:minibatch_X,Y:minibatch_Y})
        if epoch % display_epoch == 0:
            # Calculate batch loss and accuracy
            loss, acc=sess.run([loss_op,accuracy],feed_dict={X:minibatch_X,Y:minibatch_Y})
            print("epoch "+str(epoch)+", Loss= "+"{:.4f}".format(loss)+", Minibatch Accuracy= "+"{:.3f}".format(acc))
            
    # Calculate accuracy for MNIST images
    print("Training Accuracy:",sess.run(accuracy,feed_dict={X:mnist.train.images,Y:mnist.train.labels}))
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={X: mnist.test.images,Y: mnist.test.labels}))

'''
Training Accuracy: 0.9358364
Testing Accuracy: 0.9261
'''