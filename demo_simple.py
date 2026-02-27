import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sam2.build_sam import build_sam2_camera_predictor

# Parameters
N = 10  # Reset estimated mask every N frames
FPS = 100  # Frames per second for display
WATCH = False  # Set to True to watch the video frames
SAVE_RESULTS = False  # Set to True to save results
mask_folder = "2025_01_29/Board_Cylinder_1/Masks"
rgb_folder = "2025_01_29/Board_Cylinder_1/RGB"

# Initialize SAM2 predictor
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

def track_object(predictor, frame, init_mask=None):
    """
    Tracks an object in a frame using SAM2 predictor.
    
    Args:
        predictor: SAM2 camera predictor instance
        frame: Input frame (BGR format)
        init_mask: Initial mask for the first frame (numpy array, uint8), None for subsequent frames
    
    Returns:
        mask: Binary mask of the tracked object (numpy array, uint8)
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Convert frame to RGB for SAM2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if init_mask is not None:
            # Initialize tracking with the provided mask
            predictor.load_first_frame(frame_rgb)
            _, _, out_mask_logits = predictor.add_new_mask(
                frame_idx=0,
                obj_id=1,
                mask=init_mask
            )
        else:
            # Continue tracking
            _, out_mask_logits = predictor.track(frame_rgb)
        
        # Convert logits to binary mask
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)[0]
        return mask * 255  # Scale to 0-255 for consistency

def run():
    # Get list of frames
    mask_frames = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
    rgb_frames = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])

    mask_numbers = set(os.path.splitext(f)[0].split('_')[-1] for f in mask_frames)
    rgb_frames = [f for f in rgb_frames if os.path.splitext(f)[0].split('_')[-1] in mask_numbers]

    mask_files = [os.path.join(mask_folder, f) for f in mask_frames]
    rgb_files = [os.path.join(rgb_folder, f) for f in rgb_frames]

    # Initialize variables
    prev_mask = None
    prev_rgb = None
    iou_scores = []
    reset_frames = []  # Track frames where mask is reset
    frame_count = 0

    for mask_file, rgb_file in tqdm(
        zip(mask_files, rgb_files),
        total=len(mask_files),
        desc="Processing frames",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ):
        # Load images
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(rgb_file)
        
        if prev_mask is None:
            prev_mask = mask
            prev_rgb = rgb
            continue
        
        if frame_count % N == 0:
            # Reset with ground truth mask
            estimated_mask = mask
            reset_frames.append(frame_count)  # Record reset frame
            # Initialize SAM2 tracking with ground truth mask
            track_object(predictor, rgb, init_mask=mask)
        else:
            # Use SAM2 to track the object
            estimated_mask = track_object(predictor, rgb)
        
        # Draw estimated mask on RGB frame
        rgb_with_mask = rgb.copy()
        contours, _ = cv2.findContours(estimated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb_with_mask, contours, -1, (0, 255, 0), 2)
        
        # Calculate IoU
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        _, estimated_mask = cv2.threshold(estimated_mask, 127, 255, cv2.THRESH_BINARY)
        intersection = np.logical_and(estimated_mask, mask).sum()
        union = np.logical_or(estimated_mask, mask).sum()
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
        
        if WATCH:
            # Display frame
            cv2.imshow('Frame with Estimated Mask', rgb_with_mask)
            if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
                break
        
        prev_mask = estimated_mask
        prev_rgb = rgb
        frame_count += 1
        
    # Clean up
    cv2.destroyAllWindows()
    return iou_scores, reset_frames

if __name__ == "__main__":
    iou_scores, reset_frames = run()
    
    if SAVE_RESULTS:
        video_name = "Board_Cylinder_1"
        result_dir = os.path.join("result", f"sam2_{N}", video_name)
        os.makedirs(result_dir, exist_ok=True)

        np.save(os.path.join(result_dir, "iou_scores.npy"), np.array(iou_scores))
        np.save(os.path.join(result_dir, "reset_frames.npy"), np.array(reset_frames))

    # Plot IoU with reset lines
    plt.plot(iou_scores)
    plt.title('IoU Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('IoU')
    for reset_frame in reset_frames:
        plt.axvline(x=reset_frame, color='r', linestyle='--', label='Mask Reset' if reset_frame == reset_frames[0] else "")
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

    # Calculate and print mean IoU
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")