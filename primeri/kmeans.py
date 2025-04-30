import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    # Preoblikovanje slike
    orig_shape = slika.shape
    slika = slika.reshape((-1, 3))
    slika = np.float32(slika)
    
    # Kriteriji za končanje algoritma
    kriteriji = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Izvedba k-means
    _, labele, centri = cv.kmeans(slika, k, None, kriteriji, iteracije, cv.KMEANS_RANDOM_CENTERS)
    
    centri = np.uint8(centri)

    # Segmentacija slike
    segmentirana_slika = centri[labele.flatten()]
    segmentirana_slika = segmentirana_slika.reshape(orig_shape)
    
    return segmentirana_slika

if __name__ == "__main__":
    k = 3
    iteracije = 10
    slika = cv.imread("../.utils/lenna.png")
    segmentirana_slika = kmeans(slika,k,iteracije)
    
    cv.imshow("K-means s {} centri (BGR)", segmentirana_slika)
    cv.waitKey(0)
    cv.destroyAllWindows()