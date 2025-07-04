from model.features import features_extraction
import kagglehub
import os
import cv2
import pickle


def main():
    # Download latest version from kaggle
    path = kagglehub.dataset_download("rawanmostafarakha/final-star-tracker-input")
    print("Path to dataset files:", path)

    image_files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
    print(f"Found {len(image_files)} images.")

    # img_path = os.path.join(path, image_files[0])
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # features = features_extraction(img, fov_deg=20)

    # print("Number of stars:", features["num_stars"])
    # print("Centroids:\n", features["centroids"])
    # print("Brightness:\n", features["brightness"])
    # print("Sample triangle:\n", features["triangle_features"][:1])

    all_features = []
    for file in image_files:
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feats = features_extraction(img)
        feats["image_name"] = file
        all_features.append(feats)

    with open("star_features.pkl", "wb") as f:
        pickle.dump(all_features, f)

    # with open("star_features.pkl", "rb") as f:
    #   data = pickle.load(f)


            
        
