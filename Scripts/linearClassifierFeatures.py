#!/usr/bin/python
from scipy.spatial import distance
from linearClassifier import linearClassifier
from poseplot import poseSequence2D
import numpy as np
import pdb

"""
x: 1 x num pose points x num frames x 2
"""
# get first order approximation of velocity
def averagePointVelocity(x):
    phi = np.sum((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2, axis=3)       # (x1 - x2) ** 2 + (y1 - y2) ** 2
    phi = np.sqrt(phi)                                                  # get magnitude of velocity
    phi = np.mean(phi, axis=2)                                          # take average velocity of a point
    phi = np.sum(phi, axis=1)                                           # sum average point velocities
    phi = np.array([phi])                                               # format the axes

def l2ToTarget(x, target_pos):
    """
    calculates l2 distance from body points to target and sums them
    currently using: shoulder, elbow, wrist
    """
    #2-4 are pose points for right arm
    #5-7 are pose points for left arm
    right_arm = x[:, 2:5, :, :]
    left_arm = x[:, 5:8, :, :]
    l2_right = np.linalg.norm(right_arm - target_pos, axis=3) # numrows are the number of body points and number of columns are the number of frames for each body point
    l2_left = np.linalg.norm(left_arm - target_pos, axis=3)
    return l2_right, l2_left

def averagePointAcceleration(x):
    """
    calculates average acceleration for each point
    """
    velocity = x[:, :, :-1, :] - x[:, :, 1:, :]
    acceleration = velocity[:, :, :-1, :] - velocity[:, :, 1:, :]
    phi = np.sum((acceleration) ** 2, axis=3)                           # (x1 - x2) ** 2 + (y1 - y2) ** 2
    phi = np.sqrt(phi)                                                  # get magnitude of acceleration
    phi = np.mean(phi, axis=2)                                          # take average acceleration of a point
    phi = np.sum(phi, axis=1)                                           # sum average point acceleration
    phi = np.array([phi])                                               # format the axes

# extract joint angles  from body pose
# returns joint time history array N x 4 x num frames
# of [[[left shoulder angle (pts 3->2->5)], [right shoulder angle], [left elbow (pts 2, 3, 4)], [right elbow]] | ti]
def getPoseAngles(x):
    # initialize angles
    N = x.shape[0]
    T = x.shape[2]
    ang = np.zeros((N, 4, T))
    # create tuple of joint indices corresponding to LS, RS, LE, RE
    joint_ind = ((3, 2, 5), (6, 5, 2), (4, 3, 2), (7, 6, 5))
    joint_offset = (0, 0, -np.pi, -np.pi)
    # for each time step
    for i in range(T):
        # for each joint angle
        for j in range(len(joint_ind)):
            # extract the (x, y) coordinates for each associated joint
            coord = np.vstack([x[:, joint_ind[j][0], i, :], x[:, joint_ind[j][1], i, :], x[:, joint_ind[j][2], i, :]])
            # get the vectors corresponding to the coordinates
            vect = np.array([coord[0, :] - coord[1, :], coord[2, :] - coord[1, :]])
            # find the angle between them
            ang[:, j, i] = np.arccos(np.sum(vect[0, :] * vect[1, :]) / (np.linalg.norm(vect[0, :]) * np.linalg.norm(vect[1, :])))
            # offset the angle
            ang[:, j, i] += joint_offset[j]
    return ang

if __name__ == '__main__':
    keypointFolder = '/home/ian/openpose/output/keypoints/'
    keypointName = '1'
    numFrames = 1
    pose = poseSequence2D(keypointFolder, keypointName, numFrames=44, hand=True)
    x = pose.linearClassifierFormatPose()
    ang = getPoseAngles(x)
    print ang
