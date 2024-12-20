import streamlit as st

def readme():
    st.markdown("""
    # __How does it work__?           
    ## Instructions 
    - The model will predict the keypoints on the rivet.
    - The measurements will be displayed based on the keypoints.
    ## Model
    - The model is a U-Net architecture pre-trained on synthetic data and finetuned on µCT data.
    - The model predicts 6 keypoints on the rivet.
    - The keypoints are used to calculate measurements.
    ## Measurements
    - *Head Height*: Vertical distance between the first and second (:red[**red**]) keypoints.
    - *Interlock*: Horizontal distance between the second and third (:green[**green**]) keypoints.
    - *Minimum Bottom Thickness*: Distance between the third and fourth (:blue[**blue**]) keypoints.
    ## Units
    - To convert unit from pixel millimeters, please input the pixel size of the image.
    - If the pixel size is not provided, the measurements will be in pixel unit.
    ## Note
    - The model may not be accurate in all cases.
    - The measurements are based on the predicted keypoints.
    - The measurements are approximate and may not be accurate.
    - This is a demo app and should not be used for critical applications.
    - The app is for demonstration purposes only.
    - For faster processing, please use a machine with a GPU.           
                """)