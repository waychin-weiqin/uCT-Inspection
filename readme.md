# Rivet Inspection Web Application

This application is designed to perform automated visual inspection for micro-CT of rivets using deep learning. The model predicts keypoints on the rivet and provides measurements based on these keypoints.

## Overview

The application uses a U-Net architecture model pre-trained on synthetic data and fine-tuned on µCT data. The model predicts six keypoints on the rivet, which are used to calculate various measurements.

## Demo
![](./asset/demo.gif)

## Features

- Predicts keypoints on the rivet.
- Provides measurements based on the keypoints.
- Displays the keypoints and measurements on the uploaded image.
- Allows custom pixel size input for measurements.

## Instructions

### Setting Up the Application

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/waychin-weiqin/uCT-Inspection.git
    cd https://github.com/waychin-weiqin/uCT-Inspection.git
    ```

2. **Install Dependencies:**
    Make sure you have Python installed. Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Model Checkpoint:**
    Ensure you have the model checkpoint file (`models.pth`) in the `./checkpoints/` directory.

### Running the Application

1. **Start the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

2. **Access the Application:**
    Open your web browser and go to `http://localhost:8501` to access the application.

### Using the Application

1. **Upload an Image:**
    - Use the sidebar to upload an image for inspection.
    - The image should be in `jpg`, `jpeg`, or `png` format.
    - You can use the sample images included in this repository.

2. **Set Measurement Scale:**
    - Optionally, input the pixel size of the image for measurements.
    - If not provided, measurements will be in pixel units.

3. **View Results:**
    - The application will display the input image with overlaid keypoints.
    - Measurements will be shown in a table.

## Notes

- The model may not be accurate in all cases.
- Measurements are approximate and based on predicted keypoints.
- This is a demo app and should not be used for critical applications.
- For faster processing, use a machine with a GPU.

## Contact

For any issues or questions, please contact [wei.qin.chuah@rmit.edu.au].
