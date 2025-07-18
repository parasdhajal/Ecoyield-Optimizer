# Importing libraries
import os
import io
import pickle
import torch
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, session
from torchvision import transforms
from PIL import Image
from markupsafe import Markup
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import csv

# Importing custom modules
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9
import config

# Set up Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# Setting Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the CSV file for user data
USER_CSV_PATH = os.path.join(BASE_DIR, 'users.csv')

# Function to read users from the CSV file
def read_users_from_csv():
    users_list = {}
    if os.path.exists(USER_CSV_PATH):
        with open(USER_CSV_PATH, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                username, hashed_password = row
                users_list[username] = hashed_password
    return users_list

# Function to write a new user to the CSV file
def write_user_to_csv(username, hashed_password):
    with open(USER_CSV_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, hashed_password])

# Loading Models

# Plant Disease Model
disease_model_path = os.path.join(BASE_DIR, 'models', 'plant_disease_model.pth')
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Initialize model
disease_model = ResNet9(3, len(disease_classes))
if os.path.exists(disease_model_path):
    disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
    disease_model.eval()
    print("[INFO] Disease model loaded successfully.")
else:
    print(f"[ERROR] Disease model not found at {disease_model_path}")
    disease_model = None

# Predict Image
def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

def is_leaf_image(img_bytes):
    """
    Simple heuristic: checks if the image has enough green pixels to be considered a leaf.
    This is a placeholder for a real classifier.
    """
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image = image.resize((128, 128))
        arr = np.array(image)
        # Count green-dominant pixels
        green_pixels = np.sum((arr[:,:,1] > arr[:,:,0]) & (arr[:,:,1] > arr[:,:,2]) & (arr[:,:,1] > 80))
        total_pixels = arr.shape[0] * arr.shape[1]
        green_ratio = green_pixels / total_pixels
        return green_ratio > 0.2  # At least 20% green pixels
    except Exception as e:
        print(f"Error in leaf check: {e}")
        return False

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the user already exists in the CSV file
        users = read_users_from_csv()
        if username in users:
            return render_template('register.html', error="Username already exists.")
        
        # Hash the password and store the user in CSV
        hashed_password = generate_password_hash(password)
        write_user_to_csv(username, hashed_password)
        
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Read users from the CSV file
        users = read_users_from_csv() 
        
        # Check if username exists and verify password
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Public home page
@app.route('/')
def home():
    return render_template('index.html', title='EcoYield Optimizer - Home')

@app.route('/crop-recommend')
@login_required
def crop_recommend():
    return render_template('crop.html', title='EcoYield Optimizer - Crop Recommendation')

@app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='EcoYield Optimizer - Fertilizer Suggestion')

@app.route('/disease')
@login_required
def disease():
    return render_template('disease.html', title='EcoYield Optimizer - Disease Detection')

      
@app.route('/fertilizer-predict', methods=['POST'])
@login_required
def fert_recommend():
    try:
        crop_name = str(request.form['cropname']).strip().lower()
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])

        df = pd.read_csv(r'C:\\Users\\Admin\\Desktop\\Mini Project\\EcoYield Optimizer\\Data-raw\\Fertilizer.csv')
        df['Crop'] = df['Crop'].str.strip().str.lower()

        crop_data = df[df['Crop'] == crop_name]
        if crop_data.empty:
            return render_template('try_again.html', title='Fertilizer Suggestion')

        nr = crop_data['N'].iloc[0]
        pr = crop_data['P'].iloc[0]
        kr = crop_data['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]

        key = f"{max_value}{'High' if eval(max_value.lower()) < 0 else 'low'}"
        response = Markup(str(fertilizer_dic.get(key, "No recommendation available")))

        return render_template('fertilizer-result.html', recommendation=response, title='Fertilizer Suggestion')
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return render_template('try_again.html', title='Fertilizer Suggestion')

@app.route('/disease-predict', methods=['GET', 'POST'])
@login_required

def disease_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title='Harvestify - Disease Detection')

        try:
            img = file.read()
            # Leaf image verification step
            if not is_leaf_image(img):
                return render_template('disease.html', title='Harvestify - Disease Detection', alert="Invalid image uploaded")
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title='Harvestify - Disease Detection')
        except Exception as e:
            print(f"Error in prediction: {e}")
    
    return render_template('disease.html', title='Harvestify - Disease Detection')

# ==========================================================================================

if __name__ == '__main__':
    app.run(debug=True, port=5001)
