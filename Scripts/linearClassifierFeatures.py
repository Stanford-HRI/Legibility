#!/usr/bin/bash

# get first order approximation of velocity
def averagePointVelocity(x):
    phi = np.sum((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2, axis=3)       # (x1 - x2) ** 2 + (y1 - y2) ** 2
    phi = np.sqrt(phi)                                                  # get magnitude of velocity
    phi = np.mean(phi, axis=2)                                          # take average velocity of a point
    phi = np.sum(phi, axis=1)                                           # sum average point velocities
    phi = np.array([phi])                                               # format the axes
