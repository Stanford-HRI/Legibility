#!/usr/bin/python

from linearClassifier import linearClassifier
from poseplot import poseSequence2D
import numpy as np

def addFeatures(LC):
    LC.addFeature(lambda x: np.array([np.sum(np.mean(np.sqrt(np.sum((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2, axis=3)), axis=2), axis=1)]))


if __name__ == "__main__":
    keypointFolder = '/home/ian/openpose/output/keypoints/'
    keypointName = '1'
    numFrames = 1
    pose = poseSequence2D(keypointFolder, keypointName, numFrames=44, hand=True)
    x = pose.linearClassifierFormatPose()
    y = np.ones(1,)
    LC = linearClassifier(weightInitializer=0.01)
    addFeatures(LC)
    LC.train(x, y, verbose=True)
