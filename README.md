AerialVision: Drone-based Object Detection

Project Overview

AerialVision is a cutting-edge deep learning project that brings precision and intelligence to drone technology. Built to tackle the unique challenge of object detection from an aerial perspective, it transforms raw drone footage into actionable data.
At its core is a custom-trained YOLOv8 deep neural network. This model processes an entire image in a single pass, instantly identifying and localizing objects like cars, people, and bicycles. This capability is a game-changer for applications ranging from smart city planning to critical surveillance.
The system is wrapped in a crisp, interactive interface powered by Gradio, making the complexity of deep learning accessible to everyone. Simply drag, drop, and let the model work its magic.

Key Features

1.Deep Learning Model: A robust YOLOv8 model trained on the VisDrone dataset for high-accuracy object detection from an aerial perspective.
2.Intuitive UI: A clean, interactive web interface built with Gradio for seamless image uploads and result display.
3.Scalable Solution: The modular codebase allows for easy future enhancements, such as fine-tuning the model or adding new features.
4.Automated Preprocessing: A dedicated script (prepare_data.py) handles dataset conversion to the required YOLO format.

How to Run the Project
Follow these steps to set up and run the project on your local machine.

1. Clone the Repository
First, clone this repository to your local machine.

Bash-
git clone https://github.com/your-username/AerialVision.git
cd AerialVision

2. Set Up the Environment
Install the required Python libraries.

Bash-
pip install -r requirements.txt


3. Run the Application
The web application is powered by Gradio. You can start it directly from the terminal.

Bash-
python app.py
After running this command, open the provided local URL in your web browser (usually http://127.0.0.1:7860).

Project Structure
├── datasets/
│   ├── VisDrone2019-DET-train/
|   ├── VisDrone2019-DET-test-dev/
│   └── VisDrone2019-DET-val/
├── runs/
├── venv/
├── AerialVision-YOLO-format/
├── .gitignore
├── app.py
├── best.pt
├── predict.py
├── prepare_data.py
├── README.md
├── requirements.txt
└── visdrone.yaml
Note: The datasets/ and runs/ folders are not included in the repository due to their large size. You will need to download the VisDrone dataset separately and run the prepare_data.py script to generate the required AerialVision-YOLO-format folder.