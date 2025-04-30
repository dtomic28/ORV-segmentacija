import cv2 as cv
import numpy as np

def meanshift(slika):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    slika = cv.pyrMeanShiftFiltering(slika, 10, 10)