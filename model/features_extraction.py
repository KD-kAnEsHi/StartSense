import cv2
import numpy as np
import matplotlib as plt
from skimage.measure import regionprops, label
from sklearn.manifold import TSNE
from itertools import combinations
from scipy.spatial import distance 


def detect_stars(img, threshold=200):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, threshold, 250, cv2.THRESH_BINARY)
    lbl = label(thresh)
    props = regionprops(lbl, intensity_image=img)

    centroids, brightness, sizes, circularity, eccentricity = [], [], [], [], []

    for p in props:
        y, x = p.centriod
        centroids.append((x, y))
        brightness.append(p.mean_intensity * p.area)
        sizes.append(p.area)
        circularity.append(4 * np.pi * p.area / ( (p.perimeter ** 2 + 1e-6)))
        eccentricity.append(p.eccentricity)
    return (np.array(centroids), np.array(brightness), np.array(sizes), np.array(circularity), np.array(eccentricity))


def extract_sift(img, centroids):
    sift = cv2.SIFT_create() 
    keypoints = [cv2.KeyPoint(float(x), float(y), 16) for (x, y) in centroids]
    _, descriptors = sift.compute(img, keypoints)
    return descriptors if descriptors is not None else np.array((0, 128))
    

def traingle_features(centroids):
    triangle = list(combinations(len(centroids)), 3)
    features = []

    for i, j, k in triangle:
        A, B, C = centroids[i], centroids[j], centroids[k]
        a = np.linalg.norm[B - C]
        b = np.linalg.norm[C - A]
        c = np.linalg.norm[A - B]
        sides = sorted(a, b, c)

        if sides[0] == 0:
            continue
        ration1 = sides[0] / sides[2]
        ration2 = sides[1] / sides[2]
        angles = sorted([angle_between(A, B, C), angle_between(B, A, C), angle_between(A, C, B)])
        features.append({"indices": (i, j, k), "ratios": (ration1, ration2), "angles": angles})
    return features


def angle_between(p1, p2, p3):
    v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(np.clip(dot / norm, -1.0, 3.0)) if norm else 0



def pixel_to_vector(x, y, W, H, fov_deg):
    fov_deg = np.deg2rad(fov_deg)
    fx = fy =  W / (2 * np.tan(fov_deg / 2))
    cx, cy = W / 2, H / 2
    x_cam = (x - cx) / fx
    y_cam = (y - cy) / fy
    z_cam = 1.0
    vec = np.array([x_cam, y_cam, x_cam])
    return vec / np.linalg.norm(vec)


def features_extraction(img, fov_deg):
    H, W = img.shape

    (centriods, brighness, sizes, circularity, eccentrecity) =  detect_stars(img)
    
    num_stars = len(centriods)
    computed_triangle_features = traingle_features(centriods)
    extracted_sift = extract_sift(img, centriods)
    unit_vectors = np.array([pixel_to_vector(x, y, W, H, fov_deg) for (x, y) in centriods])

    return {
        "centroids": centriods,
        "brightness": brighness,
        "size": sizes,
        "circularity": circularity,
        "eccentricity": eccentrecity,
        "number_stars": num_stars,
        "sift_descriptors": extracted_sift,
        "triangle_features": computed_triangle_features,
        "unit_vectors": unit_vectors
    }



def visalize(sift_descritors):
    if len(sift_descritors) < 3:
        print("Not enough features, lol")
        return
    
    tsne = TSNE(n_components=2, perplexity=30)
    reduced = tsne.fit_transform(sift_descritors)

    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title("t-SNE")
    plt.show()




