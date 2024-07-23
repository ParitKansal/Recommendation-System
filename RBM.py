"""
@author: Parit Kansal
"""
import numpy as np
import tensorflow as tf

class RBM(object):

    def __init__(self, visibleDimensions, epochs=20, hiddenDimensions=50, ratingValues=10, learningRate=0.001, batchSize=100):
        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.ratingValues = ratingValues
        self.learningRate = learningRate
        self.batchSize = batchSize

        self._initialize_variables()

    def _initialize_variables(self):
        # Initialize weights randomly
        maxWeight = -4.0 * np.sqrt(6.0 / (self.hiddenDimensions + self.visibleDimensions))
        self.weights = tf.Variable(tf.random.uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight), dtype=tf.float32)
        self.hiddenBias = tf.Variable(tf.zeros([self.hiddenDimensions], dtype=tf.float32))
        self.visibleBias = tf.Variable(tf.zeros([self.visibleDimensions], dtype=tf.float32))

    def Train(self, X):
        optimizer = tf.optimizers.Adam(learning_rate=self.learningRate)

        for epoch in range(self.epochs):
            np.random.shuffle(X)
            for i in range(0, X.shape[0], self.batchSize):
                batch = X[i:i+self.batchSize]
                self._train_step(batch, optimizer)
            print("Trained epoch ", epoch)

    @tf.function
    def _train_step(self, batch, optimizer):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(batch)
        gradients = tape.gradient(loss, [self.weights, self.hiddenBias, self.visibleBias])
        optimizer.apply_gradients(zip(gradients, [self.weights, self.hiddenBias, self.visibleBias]))

    def _compute_loss(self, X):
        # Forward pass
        hProb0 = tf.nn.sigmoid(tf.matmul(X, self.weights) + self.hiddenBias)
        hSample = tf.nn.relu(tf.sign(hProb0 - tf.random.uniform(tf.shape(hProb0))))
        forward = tf.matmul(tf.transpose(X), hSample)

        # Backward pass
        v = tf.matmul(hSample, tf.transpose(self.weights)) + self.visibleBias
        vMask = tf.sign(X)
        vMask3D = tf.reshape(vMask, [tf.shape(v)[0], -1, self.ratingValues])
        vMask3D = tf.reduce_max(vMask3D, axis=[2], keepdims=True)
        v = tf.reshape(v, [tf.shape(v)[0], -1, self.ratingValues])
        vProb = tf.nn.softmax(v * vMask3D)
        vProb = tf.reshape(vProb, [tf.shape(v)[0], -1])
        hProb1 = tf.nn.sigmoid(tf.matmul(vProb, self.weights) + self.hiddenBias)
        backward = tf.matmul(tf.transpose(vProb), hProb1)

        # Compute divergence
        loss = tf.reduce_mean(tf.square(X - vProb))
        return loss

    def GetRecommendations(self, inputUser):
        hidden = tf.nn.sigmoid(tf.matmul(inputUser, self.weights) + self.hiddenBias)
        visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.visibleBias)
        return visible.numpy()[0]

# Usage example
# rbm = RBM(visibleDimensions=1000, epochs=20, hiddenDimensions=50, ratingValues=10, learningRate=0.001, batchSize=100)
# rbm.Train(training_data)
# recommendations = rbm.GetRecommendations(user_input)
