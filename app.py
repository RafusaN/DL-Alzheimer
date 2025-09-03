# Importing necessary libraries from Flask framework and other dependencies
from flask import Flask, request, render_template, redirect, url_for  # Flask framework for creating web app
import numpy as np  # NumPy for array manipulations
from keras.preprocessing import image  # Keras image preprocessing utilities
import tensorflow as tf  # TensorFlow for deep learning functionalities
from tensorflow.keras.models import load_model  # Loading pre-trained models in TensorFlow
import tensorflow_addons as tfa  # TensorFlow Addons for additional metrics (F1 score)
from io import BytesIO  # BytesIO to handle file streams in memory
import base64  # Base64 for encoding images for HTML display

# Initializing the Flask application
app = Flask(__name__)

# Defining the function to load a saved model with custom metrics
def load_model_with_custom_metrics():
    # Setting custom metrics to include F1 score for 4 classes, with macro averaging
    # Macro is a method for calculating evaluation metrics for multi-class classification problems. It treats all classes equally, regardless of their size.
    custom_objects = {"F1Score": tfa.metrics.F1Score(num_classes=4, average='macro')}  
    # Loading the model with custom metrics applied
    return load_model('oasis_slice_cnn_model_01.h5', custom_objects=custom_objects)

# Loading the trained model at the start of the application
model = load_model_with_custom_metrics()

# Defining the route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Checking if the request method is POST (i.e., a file was submitted)
    if request.method == 'POST':
        # Retrieving the uploaded file from the request
        file = request.files.get('file', None)
        # Checking if a file was uploaded and if it has an allowed format
        if file and allowed_file(file.filename):
            # Preprocessing the uploaded image and making a prediction
            img = preprocess_image(file)
            prediction = model.predict(img)
            # Defining class labels for model predictions
            class_names = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
            # Getting the class label with the highest prediction probability
            predicted_class = class_names[np.argmax(prediction)]
            
            # Converting the file to Base64 format for embedding in HTML
            file.stream.seek(0)  # Resetting file pointer to the beginning of the file
            base64_data = base64.b64encode(file.read()).decode('ascii')
            file_data = f"data:image/jpeg;base64,{base64_data}"

            # Rendering the index.html template with the prediction and image data
            return render_template('index.html', prediction=predicted_class, file_data=file_data)
        else:
            # If the file is not valid, display an error message on the page
            return render_template('index.html', prediction='Invalid file or format.', file_data=None)
    else:
        # If the request is GET, load the page with no data (initial load)
        return render_template('index.html', prediction=None, file_data=None)

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    # Validating that the file has an extension and is in an allowed format (PNG, JPG, JPEG)
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

# Function to preprocess the uploaded image to match the model's input requirements
def preprocess_image(file_stream):
    # Converting the uploaded FileStorage object into a BytesIO stream
    img_bytes = BytesIO(file_stream.read())
    
    # Loading the image from BytesIO and resizing it to the model’s expected input dimensions
    img = image.load_img(img_bytes, target_size=(176, 208), color_mode='rgb')
    
    # Converting the image to a NumPy array and normalizing pixel values
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Adding a batch dimension for the model input
    img /= 255.0  # Normalizing pixel values to the range [0, 1]
    
    # Returning the preprocessed image
    return img

# Defining the route for the descriptions page
@app.route('/descriptions')
def descriptions():
    # Rendering the descriptions.html template
    return render_template('descriptions.html')

# Defining the route for the paper details page
@app.route('/paper_details')
def paper_details():
    # Rendering the paper_details.html template
    return render_template('paper_details.html')

# Defining the route for the model details page
@app.route('/model_details')
def model_details():
    # Rendering the model_details.html template
    return render_template('model_details.html')

# Defining the route for the grad_cam page
@app.route('/grad_cam')
def grad_cam():
    # Rendering the grad_cam.html template
    return render_template('grad_cam.html')

# Running the app if this file is executed directly
if __name__ == '__main__':
    # Running the app in debug mode for development (shows errors in the browser)
    app.run(debug=True)
