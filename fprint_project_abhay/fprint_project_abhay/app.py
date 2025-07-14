import os
import cv2
import pickle
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}
MIN_MATCH_THRESHOLD = 50  # Minimum good matches required for a valid match

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    """
    Extract keypoints and descriptors from an image using ORB.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def find_closest_images(test_image_path, features_file):
    """
    Find all matching images to the test image based on descriptors.
    """
    # Load stored features
    with open(features_file, 'rb') as f:
        stored_features = pickle.load(f)

    # Extract features from the test image
    _, test_descriptors = extract_features(test_image_path)
    if test_descriptors is None:
        return []

    # Initialize Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_data = []

    for filename, descriptors in stored_features.items():
        # Perform KNN matching
        matches = bf.knnMatch(test_descriptors, descriptors, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # Ratio test
                good_matches.append(m)

        if len(good_matches) >= MIN_MATCH_THRESHOLD:
            matches_data.append((filename, len(good_matches)))

    # Sort matches by number of good matches
    matches_data.sort(key=lambda x: x[1], reverse=True)
    return matches_data

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the file is part of the POST request
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform prediction
            result = []
            matches = find_closest_images(filepath, features_file="features.pkl")
            if matches:
                highest = max(matches, key=lambda x: x[1])
                name_with_highest_value = highest[0]
                result = [(highest[0], highest[1])]
            #print("file path : ",filepath)

            # Pass matches to the template
            return render_template("results.html", test_image=filepath, matches=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

