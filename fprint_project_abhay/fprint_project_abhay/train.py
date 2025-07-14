import cv2
import pickle
import os

def extract_features(image_path):
    """
    Extract keypoints and descriptors from an image using ORB.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return None, None
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def save_features(image_folder, output_file):
    """
    Extract and save features (descriptors) from all images in a folder.
    """
    features = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.tif')):  # Acceptable image formats
            image_path = os.path.join(image_folder, filename)
            _, descriptors = extract_features(image_path)
            if descriptors is not None:
                features[filename] = descriptors
                print(f"Processed {filename}")
            else:
                print(f"Skipping {filename} due to missing descriptors.")
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    image_folder = 'stored_images'  # Folder containing training images
    output_file = 'features.pkl'      # File to save descriptors
    save_features(image_folder, output_file)

