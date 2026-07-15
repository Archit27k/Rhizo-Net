import cv2
import numpy as np

class MorphologyEngine:

    def __init__(self):

        self.lower_green = np.array([25, 40, 40])
        self.upper_green = np.array([90, 255, 255])

    def calculate_edge_density(self, image_path):

        image = cv2.imread(image_path)

        image = cv2.resize(image, (224, 224))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            self.lower_green,
            self.upper_green
        )

        masked = cv2.bitwise_and(
            image,
            image,
            mask=mask
        )

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 50, 150)

        edge_pixels = np.count_nonzero(edges)

        plant_pixels = np.count_nonzero(mask)

        if plant_pixels == 0:
            return 0

        density = (edge_pixels / plant_pixels) * 100

        return round(density, 2)