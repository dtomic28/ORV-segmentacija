import cv2 as cv
import numpy as np

def premikaj_prag(siva, zacetek,konec, korak):
    '''Povečaj ali zmanjšaj prag za segmentacijo slike.'''
    cv.namedWindow("Slika", cv.WINDOW_NORMAL)
    for i in range(zacetek,konec,korak):
        _, segmentirana = cv.threshold(siva, i, 255, cv.THRESH_BINARY)
        orig_seg = np.hstack((siva, segmentirana))
        cv.imshow("Slika", orig_seg)
        cv.waitKey(100)



if __name__ == "__main__":
    slika = cv.imread("../.utils/lenna.png")
    siva = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)

    print(siva.shape)

    
    _ ,segmentirana = cv.threshold(siva, 127, 255, cv.THRESH_BINARY)
    print(segmentirana.shape)

    cv.imshow("Slika", segmentirana)
    cv.waitKey(0)

    premikaj_prag(siva, 20, 240, 5)
    cv.destroyAllWindows()