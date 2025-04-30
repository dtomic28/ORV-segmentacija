import cv2 as cv
import numpy as np

def otsu(slika):
    '''Segmentacija slike z Otsujevo metodo.'''
    siva = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    _, otsu = cv.threshold(siva, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, seg = cv.threshold(siva, 127, 255, cv.THRESH_BINARY)
    orig_otsu_seg = np.hstack((siva, otsu, seg))
    cv.imshow("Otsu", orig_otsu_seg)
    cv.waitKey(0)

if __name__ == "__main__":
    slika = cv.imread("../.utils/lenna.png")
    otsu(slika)
    cv.destroyAllWindows()