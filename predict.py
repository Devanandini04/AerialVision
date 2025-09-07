from ultralytics import YOLO

# Load your custom-trained model
model = YOLO('best.pt') 

# Find a test image and place it in this folder.
# For example, name it 'test_image.jpg'.
test_image_path = 'test_image.jpg'

# Run prediction
results = model.predict(source=test_image_path, save=True, conf=0.25)

print("Prediction complete!")
print("Check the new 'runs/detect/predict' folder for the output image.")