import numpy as np
import cv2


class Harris_corner_detector(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def detect_harris_corners(self, img):
        # Smooth the image by Gaussian kernel
        img = cv2.GaussianBlur(img, (3,3), 1.5)

        # Calculate Ix, Iy (1st derivative of image along x and y axis)
        Ix = cv2.filter2D(img, -1, np.array([1., 0., -1.]).reshape(1, 3))
        Iy = cv2.filter2D(img, -1, np.array([[1.], [0.], [-1.]]))

        # Compute Ixx, Ixy, Iyy (Ixx = Ix*Ix, ...)
        Ixx = np.multiply(Ix, Ix)
        Ixy = np.multiply(Ix, Iy)
        Iyy = np.multiply(Iy, Iy)

        # Compute Sxx, Sxy, Syy (weighted summation of Ixx, Ixy, Iyy in neighbor pixels)
        Sxx = cv2.GaussianBlur(Ixx, (3,3), 1.)
        Sxy = cv2.GaussianBlur(Ixy, (3,3), 1.)
        Syy = cv2.GaussianBlur(Iyy, (3,3), 1.)

        # Compute the det and trace of matrix M (M = [[Sxx, Sxy], [Sxy, Syy]])
        det = np.multiply(Sxx, Syy) - np.multiply(Sxy, Sxy)
        trace = Sxx + Syy

        # Compute the response of the detector by det/(trace+1e-12)
        response = np.divide(det, trace+1e-12)

        return response


    def post_processing(self, response):
        # Thresholding
        response = np.where(response>self.threshold, response, 0)

        # Find local maximum
        padding_img = np.pad(response, (2, ), constant_values=(0))
        local_max = list()
        for i in range(response.shape[0]):
            for j in range(response.shape[1]):
                kernel = padding_img[i:i+5, j:j+5].reshape(-1)
                kernel[12] = -99999

                corner = True
                for x in range(25):
                    if response[i,j] <= kernel[x]:
                        corner = False
                        break

                if corner:
                    local_max.append([i,j])

        return local_max