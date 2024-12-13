import torch
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

prompt_points = []

def on_click(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        ax.plot(x, y, marker='o', color='r')
        prompt_points.append((x, y))
        plt.draw()

def create_mask_with_sam(
    image_path: str,
    model_type: str = "vit_h",
    checkpoint_path: str = "sam_vit_h.pth",
    prompt_points: np.ndarray = None,  # Array of (x, y) coordinates
    prompt_labels: np.ndarray = None,  # Array of labels
    output_mask_path: str = "mask.png"
):
    """
    Create a segmentation mask using SAM with multiple prompt points.

    :param image_path: Path to the input image
    :param model_type: Type of SAM model (vit_h, vit_l, vit_b)
    :param checkpoint_path: Path to the SAM model checkpoint
    :param prompt_points: Array of shape (N, 2) containing (x, y) coordinates
    :param prompt_labels: Array of shape (N,) containing labels (1=foreground, 0=background)
    :param output_mask_path: Where to save the binary mask
    """
    # Default single point if none provided
    if prompt_points is None:
        prompt_points = np.array([[50, 50]])
    if prompt_labels is None:
        prompt_labels = np.array([1])

    # Ensure points and labels are numpy arrays
    prompt_points = np.array(prompt_points)
    prompt_labels = np.array(prompt_labels)

    # Validate inputs
    if prompt_points.shape[0] != prompt_labels.shape[0]:
        raise ValueError("Number of points must match number of labels")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    print(f"Loaded image with shape {image.shape}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)

    # Create predictor
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # Get predictions
    masks, scores, logits = predictor.predict(
        point_coords=prompt_points,
        point_labels=prompt_labels,
        multimask_output=False
    )

    # Convert mask to image
    mask = masks[0]
    mask_img = (mask * 255).astype(np.uint8)

    # Save mask
    cv2.imwrite(output_mask_path, mask_img)
    print(f"Mask saved to {output_mask_path}")

if __name__ == "__main__":
    _image_path="./prompt3/A cat wearing wearable glasses in a watercolor style_0.jpg"
    image = Image.open(_image_path).convert("RGB")
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Connect the click event
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.title("Click and drag to draw bboxes, then close window")
    # Show the image
    plt.show()

    create_mask_with_sam(
        image_path=_image_path,
        checkpoint_path="./sam_vit_h_4b8939.pth",
        prompt_points=prompt_points,
        prompt_labels=np.array([1, 0]),
        output_mask_path="./prompt3/A cat wearing wearable glasses in a watercolor style_0_mask1.png"
    )

# wget -q 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'