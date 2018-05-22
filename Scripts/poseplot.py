#!/usr/bin/python

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess

class poseSequence2D(object):
    
    def __init__(self, keypointFolder, keypointName, numFrames, person=0, pose=True, hand=False, face=False):
        self.keypointFolder = keypointFolder
        self.keypointName = keypointName
        self.numFrames = numFrames
        self.person = person
        self._posePoints = 18
        self._handPoints = 21
        self._facePoints = 70
        self.poseData = np.zeros((self.numFrames, 2, self._posePoints))
        self.leftHandData = np.zeros((self.numFrames, 2, self._handPoints))
        self.rightHandData = np.zeros((self.numFrames, 2, self._handPoints))
        self.faceData = np.zeros((self.numFrames, 2, self._facePoints))
        self._pose = pose
        self._hand = hand
        self._face = face
        self._idxLen = 12
        self._jsonParser()
        self._poseConnections = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), 
                                  (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
                                  (12, 13), (0, 14), (14, 16), (0, 15), (15, 17)];
        
    def plotPose(self, frame=None):
        if frame is None:
            files = []
            fig, ax = plt.subplots()
            for i in range(self.numFrames):
                plt.cla()
                pltName = '_tmp%03d.png' % i
                plt.plot(self.poseData[i, 0, :], self.poseData[i, 1, :], 'bo')
                skeletonX = self._drawConnections(self._poseConnections, self.poseData[i, 0, :])
                skeletonY = self._drawConnections(self._poseConnections, self.poseData[i, 1, :])
                for j in range(len(skeletonX)):
                    plt.plot(skeletonX[j], skeletonY[j], 'r')
                plt.xlim(0, 1200)
                plt.ylim(-800, 0)
                plt.savefig(pltName)
                files.append(pltName)
            subprocess.call("mencoder 'mf://_*.png' -mf type=png:fps=30 -ovc lavc "
                            "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg", shell=True)
            subprocess.call("rm _tmp*", shell=True)
                
            
    def _jsonParser(self):
        # for every frame
        for i in xrange(self.numFrames):
            # get the file name
            fileName = self._getFileName(i)
            # open the file
            with open(fileName) as jsf:
                # load the json data
                data = json.load(jsf)
                # if there is pose data
                if self._pose:
                    # add pose data
                    dataPose = data['people'][self.person]['pose_keypoints_2d']
                    self.poseData[i, 0] = [dataPose[j * 3] for j in range(self.poseData.shape[2])]
                    self.poseData[i, 1] = [-dataPose[j * 3 + 1] for j in range(self.poseData.shape[2])]
                # if there is hand data
                if self._hand:
                    # add hand data
                    dataLeftHand = data['people'][self.person]['hand_left_keypoints_2d']
                    self.leftHandData[i, 0] = [dataLeftHand[j * 3] for j in range(self.leftHandData.shape[2])]
                    self.leftHandData[i, 1] = [dataLeftHand[j * 3 + 1] for j in range(self.leftHandData.shape[2])]
                    dataRightHand = data['people'][self.person]['hand_right_keypoints_2d']
                    self.rightHandData[i, 0] = [dataRightHand[j * 3] for j in range(self.rightHandData.shape[2])]
                    self.rightHandData[i, 1] = [dataRightHand[j * 3 + 1] for j in range(self.rightHandData.shape[2])]
                # if there is face data
                if self._face:
                    # add face data
                    dataFace = data['people'][self.person]['face_keypoints_2d']
                    self.faceData[i, 0] = [dataFace[j * 3] for j in range(self.faceData.shape[2])]
                    self.faceData[i, 1] = [dataFace[j * 3 + 1] for j in range(self.faceData.shape[2])]
                
    def _getFileName(self, idx):
        fileName = self.keypointFolder + self.keypointName + '_' + str(idx).zfill(self._idxLen) + '_keypoints.json'
        return fileName
        
    def _drawConnections(self, connections, coordinates):
        skeleton = []
        for connection in connections:
            c0 = coordinates[connection[0]]
            c1 = coordinates[connection[1]]
            if c0 != 0.0 and c1 != 0.0:
                skeleton += [(c0, c1)]
        return skeleton
        
    def linearClassifierFormatPose(self):
        # x needs to be a single row
        # lets have x be a list of time histories for each point
        # each time history will have numframe tuples of (x, y) pairs corresponding to each point
        # so the array will be 1 x numpoints x numframes x  2
        # initialize x
        x = np.zeros((1, self._posePoints, self.numFrames, 2))
        # loop over pose points
        for point in xrange(self._posePoints):
            # loop over frames
            for frame in xrange(self.numFrames):
                # set the x y coordinate
                x[0, point, frame, 0] = self.poseData[frame, 0, point]
                x[0, point, frame, 1] = self.poseData[frame, 1, point]
        return x
        
if __name__ == '__main__':
    keypointFolder = '/home/ian/openpose/output/keypoints/'
    keypointName = '1'
    pose = poseSequence2D(keypointFolder, keypointName, numFrames=44, hand=True)
    pose.plotPose()
    
