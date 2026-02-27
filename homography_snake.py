import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import color
from skimage.segmentation import active_contour
from skimage.filters import gaussian

# Parameters
N = 20  # Reset estimated mask every N frames with ground truth
M = 5   # Apply active contour correction every M frames between resets
FPS = 100  # Frames per second for display
WATCH = True  # Set to True to watch the video frames
mask_folder = "2025_01_29/Board_Cylinder_1/Masks"
rgb_folder = "2025_01_29/Board_Cylinder_1/RGB"

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
    correction_frames = []  # Track frames where active contour correction is applied
    frame_count = 0
    frames_since_reset = 0  # Counter for frames since last ground truth reset

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
            estimated_mask = mask
            reset_frames.append(frame_count)  # Record reset frame
            frames_since_reset = 0  # Reset counter
        else:
            # ORB feature detection on RGB images
            orb = cv2.ORB_create(nfeatures=1000)
            gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            gray_prev_rgb = cv2.cvtColor(prev_rgb, cv2.COLOR_BGR2GRAY)
            kp1, des1 = orb.detectAndCompute(gray_prev_rgb, None)
            kp2, des2 = orb.detectAndCompute(gray_rgb, None)
            
            # Feature matching with ratio test
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                
                # Estimate next mask
                h, w = prev_mask.shape
                if H is not None and np.linalg.cond(H) < 1e6:
                    estimated_mask = cv2.warpPerspective(prev_mask, H, (w, h))
                    _, estimated_mask = cv2.threshold(estimated_mask, 127, 255, cv2.THRESH_BINARY)
                else:
                    estimated_mask = prev_mask
            else:
                estimated_mask = prev_mask
            
            # Apply active contour correction every M frames between resets
            if frames_since_reset % M == 0 and frames_since_reset != 0:
                try:
                    # Convert RGB to grayscale and apply Gaussian smoothing
                    gray = color.rgb2gray(rgb)
                    smoothed = gaussian(gray, sigma=3)
                    
                    # Compute gradient magnitude of the smoothed image
                    grad_y, grad_x = np.gradient(smoothed)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    cv2.imshow('Gradient Magnitude', gradient_magnitude*100)  # Display gradient magnitude
                    
                    # Extract initial contour from estimated mask
                    contours, _ = cv2.findContours(estimated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Use the largest contour
                        contour = max(contours, key=cv2.contourArea)
                        init = contour.squeeze()
                        if len(init) >= 3:  # Ensure enough points for active contour
                            # Run active contour
                            snake = active_contour(
                                smoothed,
                                init,
                                alpha=0.5,  # Contour rigidity
                                beta=10,      # Contour smoothness
                                gamma=0.000001  # Step size
                                # max_iterations=100
                            )
                            # Convert snake points to a binary mask
                            estimated_mask = np.zeros_like(estimated_mask)
                            snake = snake.astype(np.int32)
                            cv2.fillPoly(estimated_mask, [snake], 255)
                            correction_frames.append(frame_count)  # Record correction frame
                except Exception as e:
                    # If active contour fails, keep the estimated mask
                    print(f"Error: {e}")
                    pass
        
        # Draw estimated mask on RGB frame
        rgb_with_mask = rgb.copy()
        contours, _ = cv2.findContours(estimated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb_with_mask, contours, -1, (0, 255, 0), 2)
        
        # Calculate IoU
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
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
        frames_since_reset += 1
    
    # Clean up
    cv2.destroyAllWindows()
    return iou_scores, reset_frames, correction_frames

if __name__ == "__main__":
    iou_scores, reset_frames, correction_frames = run()

    # Plot IoU with reset and correction lines
    plt.plot(iou_scores)
    plt.title('IoU Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('IoU')
    for reset_frame in reset_frames:
        plt.axvline(x=reset_frame, color='r', linestyle='--', label='Mask Reset' if reset_frame == reset_frames[0] else "")
    for corr_frame in correction_frames:
        plt.axvline(x=corr_frame, color='b', linestyle=':', label='Active Contour Correction' if corr_frame == correction_frames[0] else "")
    plt.legend()
    plt.show()

    # Calculate and print mean IoU
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")