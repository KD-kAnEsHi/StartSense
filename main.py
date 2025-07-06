from model.features import features_extraction
import matplotlib.pyplot as plt
import kagglehub
import os
import cv2
import pickle


def main():
    # Download latest version from kaggle
    path = os.path.join(os.path.dirname(__file__), "archive") 
    """ kagglehub.dataset_download("rawanmostafarakha/final-star-tracker-input") """
    
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
    for i, file in enumerate(image_files):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to read image {img_path}")
            continue


        feats = features_extraction(img, fov_deg=20)
        feats["image_name"] = file
        all_features.append(feats)

        # Print debug info for first image
        if i == 0:
            print("Image:", file)
            print("Number of stars:", feats["number_stars"])
            print("Centroids:", feats["centroids"][:5])  # print first 5
            print("Brightness:", feats["brightness"][:5])
            print("Triangle features:", feats["triangle_features"][:1])
            print("Unit vectors:", feats["unit_vectors"][:2])
            print("SIFT shape:", feats["sift_descriptors"].shape)

    # pickle file save
    with open("star_features.pkl", "wb") as f:
        pickle.dump(all_features, f)

    # with open("star_features.pkl", "rb") as f:
    #   data = pickle.load(f)

    # remove late: is it realling seing stars
    img = cv2.imread(os.path.join(path, image_files[0]), cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')

    for (x, y) in all_features[0]["centroids"]:
        plt.plot(x, y, 'ro', markersize=3)

    plt.title("Detected Star Centroids")
    plt.show()
        


if __name__ == "__main__":
    main()