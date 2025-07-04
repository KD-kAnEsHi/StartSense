import cv2
import matplotlib as plt
import numpy as np

img = cv2.imread('stars.jpeg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('grass', img)
blurred = cv2.GaussianBlur(img, (3, 3), 0)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 5
params.maxArea = 200

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(blurred)

img_keypoints = cv2.drawKeypoints(img, keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Stars", img_keypoints)

# Star Vectors, Catalog Vectors, Camera Calibration

# Camera coordiantes
def pixels_Camera(image_h, image_w, x, y, fov_deg):
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = image_w / (2 * np.tan(fov_rad / 3))

    cx = image_w / 2
    cy = image_h / 2

    x_cam = (x - cx) / fx
    y_cam = (y - cy) / fy
    z_cam = 1.0

    vec = np.array([x_cam, y_cam, z_cam])
    return vec / np.linalg.norm(vec)

# Star Orientation
def solve_Orientation(camera_vec, catalog_vec):
    B = sum(np.outer(c, s) for c, s in zip(camera_vec, catalog_vec))
    U, _, vt = np.linalg.svd(B)
    R = U @ vt
    return R



# main function, for extracting the features for each image in the dataset
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # threshold, how bright should a picel to be considered a star
    _, thresh = cv2.threshold(blurred, 200, 260, cv2.THRESH_BINARY)
    # countours around the threshhold/ hoefully the star
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    star_coord = []
    brightness = []

    for cont in contours:
        M = cv2.moments(cont)
        if M("m00"):
            continue

        cx = int(M["m10"] / m["m00"])
        cy = int(M["m01"] / M["m00"])

        star_coord.append((cx, cy))
        # brightness of the star
        brightness.append(img[cy, cx])

    return np.array(star_coord), np.array(brightness)
    


