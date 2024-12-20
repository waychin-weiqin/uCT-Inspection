from model import UNet
import numpy as np
import streamlit as st 
import torch 
import os 
import pandas as pd

from torchvision import transforms
from PIL import Image
from utils import get_keypoints, kp_to_measurements, overlay_keypoints
from readme import readme

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=1, out_channels=6, concat=False).to(device)
state_dict_path = "./checkpoints/models.pth" # finetuned on micro-CT
state_dict = torch.load(state_dict_path)
_ = model.load_state_dict(state_dict)
model.eval()

if "model" not in st.session_state:
    st.session_state.model = model
    st.write("Model loaded.")

transform = transforms.Compose([transforms.Resize((112, 112)),
                                transforms.ToTensor()])

if __name__ == "__main__":
    st.title("Rivet Inspection Web Application")

    # Load image from file uploader and display in the sidebar
    with st.sidebar:
        st.markdown("# Introduction")
        st.write("""
             - This is a demo app for rivet inspection using deep learning.
             - The model will predict the keypoints on the rivet and provide measurements based on the keypoints.
             - The keypoints are color-coded and displayed on the image.
             - See below for instruction.
             """)
        
        st.markdown("---")

        st.write("**Please upload an image for inspection.**")
        uploaded_file = st.file_uploader("Choose an image...", 
                                         type=["jpg", "jpeg", "png"],
                                         help="Please upload an image for inspection.",
                                         accept_multiple_files=False)
        st.markdown("---")

        # Add an input to input pixel size for measurements
        # Only allows input if the checkbox is checked
        st.write("### Measurement Scale")
        use_custom_pixel_size = st.checkbox("Use custom pixel size for measurements")
        if use_custom_pixel_size:
            st.write("Please input the pixel size of the image for measurements.")
            pixel_size = st.number_input("Pixel size (in mm):", value=1.0, step=0.1)
        else:
            pixel_size = 1.0  # Default pixel size

        st.markdown("---")
        readme()

    if not uploaded_file:
        st.write("Please upload an image on the sidebar for inspection.")
        st.stop()

    if uploaded_file is not None:
        # Perform inference
        image_raw = Image.open(uploaded_file).convert("RGB")
        image = transform(image_raw.convert("L")).unsqueeze(0).to(device)

        raw_x, raw_y = image_raw.size
        x, y = image.size()[2:]
        scales = np.array([raw_x / x, raw_y / y])

        output = model(image).squeeze(0).cpu().detach().numpy()
        keypoints = get_keypoints(output)
        measurements = kp_to_measurements(keypoints)    # in pixel units
        measurements_mm = {key: value * pixel_size for key, value in measurements.items()}  # in mm units
        overlay = overlay_keypoints(image_raw, keypoints, scales)
        overlay = Image.fromarray(overlay.squeeze()).convert("RGB")

        # Display measurements in table (measurements in columns and units in rows)
        st.write("### Estimated Measurements:")
        if not use_custom_pixel_size: # show table with pixel measurements only 
            measurements_df = pd.DataFrame([(key, value) for key, value in measurements.items()], 
                    columns=["Measurement", "Value (pixel)"])
        else:
            measurements_df = pd.DataFrame([(key, measurements[key], value) for key, value in measurements_mm.items()], 
                    columns=["Measurement", "Value (pixel)", "Value (mm)"])
        st.container().table(measurements_df)
        # Display input and output images (Put them side by side for easy visual comparison)
        # st.image([image_raw, overlay], width=200, caption=["Input Image", "Output Image"], use_container_width=False)
        # st.image(image_raw, width=200, caption="Input Image", use_container_width=False)
        # st.image(overlay, width=200, caption="Output Image", use_container_width=False)

        row = st.columns(2)
        row[0].image(image_raw, caption="Input Image", use_container_width=True)
        row[1].image(overlay, caption="Output Image", use_container_width=True)
        
    # with st.sidebar:
    #     overlay.save(os.path.join("outputs/example.png"))
    #     st.write("Output image saved to outputs/example.png")

    #     # st.write("Measurements:")
    #     # st.write(measurements)
