# generating angled grating stimuli
import numpy as np

from numpy.random import binomial, normal


def generateGrating(freq, theta, pixelDim=201, shotNoise=0., noiseVar=0.):
    """Generated angled grating with specified frequency and orientation"""
    xs = np.linspace(-np.pi, np.pi, pixelDim)
    ys = np.linspace(-np.pi, np.pi, pixelDim)

    X, Y = np.meshgrid(xs, ys)
    Z = np.cos( freq * (Y*np.cos(theta) + X*np.sin(theta) ) )

    # add noise to the generated gratings
    noiseLocations = binomial(1, shotNoise, size=(pixelDim, pixelDim))
    noiseMagnitude = normal(scale=noiseVar**0.5, size=(pixelDim, pixelDim))

    Z = np.maximum( np.minimum( Z+noiseLocations * noiseMagnitude, 1), -1)

    r2 = X**2 + Y**2
    Z[r2 >= 6] = 0.

    return Z



def searchForAngle(targetGrating, endpoints, targetResolution, makeGrating):
    """Positive control: repeatedly search for the angle with maximum overlapping grating"""
    angles = np.linspace(endpoints[0], endpoints[1], 20)
    
    overlaps = []
    for angle in angles:
        test = makeGrating(angle)
        overlaps.append( findOverlap( targetGrating, test)  )
    
    maxInd = np.argmax( np.array(overlaps) )
    
    if endpoints[1] - endpoints[0] < targetResolution:
        return angles[maxInd]
    
    nextMin = angles[ max( maxInd-1, 0) ]
    nextMax = angles[ min( maxInd+1, len(angles)-1) ]
    
    return searchForAngle(targetGrating, [ nextMin, nextMax ] , targetResolution, makeGrating)


def findOverlap(grating1, grating2):
    return np.sum(grating1*grating2)

