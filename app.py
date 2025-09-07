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