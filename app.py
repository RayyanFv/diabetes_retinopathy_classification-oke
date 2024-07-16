from flask import Flask, render_template, request, redirect, url_for, session, flash
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
import time
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Firebase Admin SDK initialization
cred = credentials.Certificate("my-untitled-project-a490d-firebase-adminsdk-h9oop-fa2bb33f60.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model/model.tflite')
interpreter.allocate_tensors()

# Function to perform inference
def predict(input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the input data if necessary
    input_data = np.array(input_data, dtype=np.float32)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/index')
def index():
    user = session.get('user')
    if not user:
        return redirect(url_for('login'))
    # Fetch user info from Firestore
    user_ref = db.collection('users').document(user['uid'])
    user_doc = user_ref.get()
    if user_doc.exists:
        user_info = user_doc.to_dict()
        username = user_info.get('username')
    else:
        username = 'Unknown User'
    return render_template('index.html', user=username)


@app.route('/analysis')
def analysis():
    user = session.get('user')
    if not user:
        return redirect(url_for('login'))
    return render_template('analysis.html', user=user)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Image processing and prediction logic here
            image = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
            input_data = tf.keras.preprocessing.image.img_to_array(image)

            # Flip the image horizontally
            input_data = tf.image.flip_left_right(input_data)

            # Expand dimensions to fit model input
            input_data = np.expand_dims(input_data, axis=0)

            # Perform prediction
            prediction = predict(input_data)

            # Convert prediction to readable format if necessary
            prediction_class = np.argmax(prediction, axis=1)
            class_labels = ['Normal', 'Non Proliferative Diabetic Retinopathy', 'Proliferative ']  # Replace with actual class labels
            predicted_label = class_labels[prediction_class[0]]
            confidence = prediction[0][prediction_class[0]]

            return render_template('upload.html', img_path=file_path, result=predicted_label, confidence=confidence)
    return render_template('upload.html')

@app.route('/send_user')
def send_user():
    return render_template('send_user.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/chat')
def chat():
    user = session.get('user')
    if not user:
        return redirect(url_for('login'))
    other_user_id = request.args.get('other_user_id')
    return render_template('chat.html', other_user_id=other_user_id, user_id=user['uid'])

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        id_token = request.form['idToken']
        try:
            time.sleep(1)  # To handle clock skew issues
            decoded_token = auth.verify_id_token(id_token)
            session['user'] = decoded_token
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            flash('Login failed: ' + str(e), 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.create_user(email=email, password=password)
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash('Registration failed: ' + str(e), 'danger')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/send_recommendation', methods=['GET', 'POST'])
def send_recommendation():
    if request.method == 'POST':
        user = request.form['user']
        recommendation = request.form['recommendation']
        warning = request.form['warning']
        date = request.form['date']
        image = request.files.get('image')

        image_url = ''
        if image:
            image_filename = f'recommendation_images/{int(time.time())}_{image.filename}'
            image.save(os.path.join(UPLOAD_FOLDER, image_filename))
            image_url = url_for('static', filename=f'uploads/{image_filename}', _external=True)

        # Save to Firestore
        db.collection('recommendations').add({
            'uid': user,
            'recommendation': recommendation,
            'date': date,
            'imageUrl': image_url
        })

        db.collection('warnings').add({
            'uid': user,
            'warning': warning
        })

        flash('Data sent successfully!', 'success')
        return redirect(url_for('send_recommendation'))
    return render_template('advice.html')

if __name__ == '__main__':
    app.run(debug=True)
