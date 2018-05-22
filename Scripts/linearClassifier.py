#!/usr/bin/python

import numpy as np

class linearClassifier(object):
    def __init__(self, weightInitializer=1.0, learningRate=1.0E-4):
        self.features=np.array([])
        self.weightInitializer = weightInitializer
        self.weights=np.array([])
        self.ltype = 'l2'
        self.learningRate = learningRate;
        
    def addFeature(self, *args):
        # get size of input
        Hi = len(args)
        # add functions to list of features
        self.features = np.hstack((self.features, np.array(args)))
        # add weights
        self.weights = np.hstack((self.weights, self.weightInitializer * np.random.randn(Hi)))
    
    def phi(self, x):
        # map the features functions to x
        phix = map(lambda f: f(x), self.features)
        phix = np.hstack(phix)
        return phix
            
    
    def score(self, x):
        # transform x: N x D into scores: N x H
        # calculate w * phi
        score = self.weights * self.phi(x)
        return score
    
    def predict(self, x):
        # get the score
        score = self.score(x)
        # for now, the prediction is the sum of the scores for an example
        yHat = np.sum(score, axis=1)
        return yHat
    
    def loss(self, x, y):
        # get predicted label
        yHat = self.predict(x)
        # l2 Loss
        if (self.ltype == 'l2'):
            loss = np.mean((yHat - y) ** 2)
        return loss
        
    def gradStep(self, x, y):
        # procedure depends on how we calculate loss
        if (self.ltype == 'l2'):
            yHat = self.predict(x)
            grad = 2.0 * np.mean(np.expand_dims((yHat - y), 1) * self.phi(x), axis=0)
            self.weights -= self.learningRate * grad
    
    def train(self, x, y, numSteps=100, verbose=False):
        for i in xrange(numSteps):
            if verbose:
                print "Step ", i, ": \n\tLoss = ", self.loss(x, y), "\n\tWeights: ", self.weights
            self.gradStep(x, y)
            
    
if __name__ == "__main__":
    # validate architecture against f(x) = a * x^2 + b * x + c
    LC = linearClassifier(weightInitializer=0.01)
    a = 0.1
    b = 4.0
    c = -10.0
    x = np.array([np.arange(-10, 10.1, 0.1)]).T
    y = np.reshape(a * x ** 2 + b * x + c, (-1))
    LC.addFeature(lambda x: x ** 2, lambda x: x ** 1, lambda x: x ** 0)
    LC.train(x, y, numSteps=1, verbose=True)
