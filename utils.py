import numpy as np 
import torch 
import os 
import math 
import cv2


def get_keypoints(predictions):
    """
    Extracts predicted keypoints from the heatmaps.

    Args:
    - predictions (torch.Tensor): Predicted heatmaps from the model. Shape: (6, H, W)

    Output:
    - keypoints (np.array): Predicted keypoints. Shape: (6, 2)
    """

    # Make sure there are 6 heatmaps
    assert predictions.shape[0] == 6

    keypoints = []
    for i in range(predictions.shape[0]):
        heatmap = predictions[i]
        pred_coords = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # Flip the coordinates
        pred_coords = (pred_coords[1], pred_coords[0])
        keypoints.append(pred_coords)
    return np.array(keypoints)


def kp_to_measurements(keypoints):
    """
    Converts keypoints to measurements.
    
    Args:
    - keypoints (np.array): Predicted keypoints (x, y). Shape: (6, 2)

    Output:
    - measurements (np.array) Shape: (3): 
        Head height: vertical distance between first and second keypoints
        Interlock: horizontal distance between second and third keypoints
        Minimum Bottom Thickness: distance between third and fourth keypoints
    """

    # Define the measurements
    head_height = keypoints[0][1] - keypoints[1][1]
    interlock = keypoints[2][0] - keypoints[3][0]
    min_bottom_thickness = np.linalg.norm(keypoints[4] - keypoints[5])

    return {"head_height": float(-head_height), 
            "interlock": float(-interlock), 
            "min_bottom_thickness": min_bottom_thickness}


def overlay_keypoints(input_images, keypoints, scales):
    """
    Overlays keypoints on the input images. 
    The keypoints are required to scale back to the original image size.
    Use color to indicate the keypoints:
        - Red: KP1, KP2
        - Green: KP3, KP4
        - Blue: KP5, KP6
    Args:
    - input_images (PIL Image): Input images. Shape: (H, W, 3)
    - keypoints (np.array): Predicted keypoints. Shape: (6, 2)
    - scales (np.array): Scales for resizing the image. Shape: (N)
    
    Output:
    - output_images (np.array): Images with keypoints overlaid. Shape: (N, H, W, C)
    """
    # Convert input image to numpy array
    input_images = np.array(input_images)

    # Scale keypoints back to the original image size
    keypoints = keypoints * scales[None, :]    

    output_images = input_images.copy()
    colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)]
    
    for i in range(len(keypoints)):
        x, y = keypoints[i]
        x, y = int(x), int(y)
        output_images = cv2.circle(output_images, (x, y), 3, colors[i], -1)
    
    return output_images

def pixel_to_mm(measurements, pixel_size=0.1):
    """
    Converts pixel measurements to mm.

    Args:
    - measurements (dict): Measurements in pixels.
    - pixel_size (float): Pixel size in mm.

    Output:
    - measurements_mm (dict): Measurements in mm.
    """
    measurements_mm = {}
    for key, value in measurements.items():
        measurements_mm[key] = value * pixel_size
    return measurements_mm