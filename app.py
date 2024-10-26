from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from io import BytesIO
import base64

app = Flask(__name__)

# Defining the saved model path 
def load_model_with_custom_metrics():
    custom_objects = {"F1Score": tfa.metrics.F1Score(num_classes=4, average='macro')}  
    return load_model('oasis_slice_cnn_model_01.h5', custom_objects=custom_objects)

# Loading the trained model
model = load_model_with_custom_metrics()

# Home page route
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file', None)
        if file and allowed_file(file.filename):
            # Process the image and make a prediction
            img = preprocess_image(file)
            prediction = model.predict(img)
            class_names = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
            predicted_class = class_names[np.argmax(prediction)]
            
            # Convert file to Base64 for displaying
            file.stream.seek(0)  # Go to the beginning of the file
            base64_data = base64.b64encode(file.read()).decode('ascii')
            file_data = f"data:image/jpeg;base64,{base64_data}"

            return render_template('index.html', prediction=predicted_class, file_data=file_data)
        else:
            # Handle cases where the file is not valid
            return render_template('index.html', prediction='Invalid file or format.', file_data=None)
    else:
        # Initial page load with no data
        return render_template('index.html', prediction=None, file_data=None)
    

# Checking if file extension is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']


# Preprocessing the image to fit the model's input requirements
def preprocess_image(file_stream):
    # Converting the FileStorage object to BytesIO
    img_bytes = BytesIO(file_stream.read())
    
    # Loading the image from BytesIO object
    img = image.load_img(img_bytes, target_size=(176, 208), color_mode='rgb')
    
    # Converting the image to a numpy array and normalize it
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add the batch dimension
    img /= 255.0  # Normalize to [0,1]
    
    return img

# Description page route
@app.route('/descriptions')
def descriptions():
    return render_template('descriptions.html')

# paper_details page route
@app.route('/paper_details')
def paper_details():
    return render_template('paper_details.html')

# model_details page route
@app.route('/model_details')
def model_details():
    return render_template('model_details.html')

# grad_cam page route
@app.route('/grad_cam')
def grad_cam():
    return render_template('grad_cam.html')


if __name__ == '__main__':
    app.run(debug=True)

