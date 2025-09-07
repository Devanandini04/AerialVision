# import os
# from flask import Flask, request, render_template, send_from_directory
# from ultralytics import YOLO
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # --- Configuration ---
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # The results will be saved in the default 'runs/detect' folder by YOLO

# # Load your custom-trained YOLO model
# model = YOLO('best.pt')

# @app.route('/', methods=['GET', 'POST'])
# def predict_image():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return "No file part"
#         file = request.files['file']
#         if file.filename == '':
#             return "No selected file"
        
#         if file:
#             # Save the uploaded file to the 'uploads' folder
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(UPLOAD_FOLDER, filename)
#             file.save(filepath)

#             # Run YOLO prediction
#             # The 'project' and 'name' arguments control the output directory
#             # This will save results to 'runs/detect/predictions'
#             model.predict(filepath, save=True, project='runs/detect', name='predictions', exist_ok=True)
            
#             # The result image will have the same name as the uploaded file
#             result_image_path = os.path.join('runs/detect/predictions', filename)
            
#             return render_template('result.html', image_name=result_image_path)
    
#     # Show the upload form on GET request
#     return render_template('index.html')


# # Route to display the image
# @app.route('/display/<path:filename>')
# def display_image(filename):
#     return send_from_directory('.', filename, as_attachment=False)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
import gradio as gr
from ultralytics import YOLO
import os

# Load your custom-trained YOLO model
model = YOLO('best.pt')

# Define the function that Gradio will use to run predictions
def detect_objects(image):
    """
    Takes an image path, runs object detection, and returns the annotated image.
    """
    if image is None:
        return "Please upload an image."

    # Predict on the input image
    # The 'save=True' argument tells YOLO to save the annotated image
    results = model.predict(source=image, save=True, conf=0.25)
    
    # Get the path to the saved result image from the results object
    if results and len(results) > 0:
        # The YOLO library saves the annotated image inside a new directory within 'runs/detect'
        # We need to get the path to this directory
        result_dir = results[0].save_dir
        
        # The annotated image name is the same as the base name of the input image
        annotated_image_path = os.path.join(result_dir, os.path.basename(image))
        return annotated_image_path
    else:
        return "Error: Could not perform prediction."

# Create the Gradio interface
# We define the function to call, the input component, and the output component
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="filepath", label="Upload Drone Image"),
    outputs=gr.Image(type="filepath", label="Detection Result"),
    title="Drone Object Detection",
    description="Upload an image and the model will detect pedestrians, cars, and other objects."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()