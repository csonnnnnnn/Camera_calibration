import cv2
import numpy as np
import glob
import json

number_of_squares_x = 11
number_of_internal_corners_x = number_of_squares_x - 1
number_of_squares_y = 8
number_of_internal_corners_y = number_of_squares_y - 1
square_size = 0.023 # in meters
frame_interval = 30

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)

videos = glob.glob('videos/*.mp4')  # Update with your video directory path

cap = cv2.VideoCapture(videos[0])
if not cap.isOpened():
    print(f"Error: Could not open video {videos[0]}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Dimensions: Width = {width}, Height = {height}")
cap.release()

camera_name = ['Wide cam', 'Cam0', 'Cam1', 'Cam2', 'Cam3']

# Automatically calculate ROIs for a 2x3 grid
cell_width = width // 3 
cell_height = height // 2  
rois = [
    (2 * cell_width, 0, cell_width, cell_height),       # Camera wide: (1,3) Top-right
    (0, cell_height, cell_width, cell_height),          # Camera 0: (2,1) Bottom-left
    (cell_width, cell_height, cell_width, cell_height),  # Camera 1: (2,2) Bottom-middle
    (0, 0, cell_width, cell_height),                    # Camera 2: (1,1) Top-left
    (cell_width, 0, cell_width, cell_height)           # Camera 3: (1,2) Top-middle
]
print("Calculated ROIs for 2x3 grid (x, y, width, height):")
for i, roi in enumerate(rois):
    print(camera_name[i], f": {roi}")

# cap = cv2.VideoCapture(videos[0])
# ret, frame = cap.read()
# if ret:
#     for cam_idx, (x, y, w, h) in enumerate(rois):
#         sub_image = frame[y:y+h, x:x+w]
#         cv2.imshow("ROI of " + camera_name[cam_idx], sub_image)
#         cv2.waitKey(0)
# cv2.destroyAllWindows()
# cap.release()

# Initialize lists for each camera
num_cameras = 5
objpoints = [[] for _ in range(num_cameras)]  # 3D points for each camera
imgpoints = [[] for _ in range(num_cameras)]  # 2D points for each camera
frame_indices = [[] for _ in range(num_cameras)]  # Track frame indices for stereo calibration

# Process each video for chessboard corner detection
for video_idx, video_path in enumerate(videos):
    print(f"\nProcessing Video {video_idx + 1}: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        continue

    # Verify video dimensions match the first video
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if vid_width != width or vid_height != height:
        print(f"Warning: Video {video_path} has different dimensions: Width = {vid_width}, Height = {vid_height}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            for cam_idx, roi in enumerate(rois):
                x, y, w, h = roi
                sub_image = frame[y:y+h, x:x+w]
                gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)

                ret, corners = cv2.findChessboardCorners(gray, (number_of_internal_corners_x,number_of_internal_corners_y), None)

                if ret:
                    objpoints[cam_idx].append(objp)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints[cam_idx].append(corners)
                    frame_indices[cam_idx].append((video_idx, frame_count))

                    cv2.drawChessboardCorners(sub_image, (number_of_internal_corners_x,number_of_internal_corners_y), corners, ret)
                    frame[y:y+h, x:x+w] = sub_image
                    cv2.imshow(f'Video {video_idx + 1} ' + camera_name[cam_idx] +  ' corners', sub_image)
                    cv2.waitKey(1)
        frame_count += 1
    cap.release()
cv2.destroyAllWindows()

# Perform camera calibration for each camera
intrinsics = []
dist_coeffs = []
for cam_idx in range(num_cameras):
    print(f"\nProcessing", camera_name[cam_idx], "...")
    if len(objpoints[cam_idx]) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints[cam_idx], imgpoints[cam_idx], gray.shape[::-1], None, None
        )
        intrinsics.append(mtx)
        dist_coeffs.append(dist)

        print(camera_name[cam_idx], f"Intrinsic Matrix:\n", mtx)
        print(camera_name[cam_idx], f"Distortion Coefficients:\n", dist)
        print(camera_name[cam_idx], f": {len(rvecs)} extrinsic matrices (relative to chessboard)")
    else:
        print(f"No valid chessboard corners found for ", camera_name[cam_idx])
        intrinsics.append(None)
        dist_coeffs.append(None)

# Save extrinsic parameters for each frame
relative_extrinsics = []
extrinsic_matrix = [[] for _ in range(num_cameras)]
extrinsic_matrix[0] = np.eye(4)

for cam_idx in range(1, num_cameras):  # Cameras 0, 1, 2, 3
    print(f"\nComputing relative pose between Wide cam and", camera_name[cam_idx], "...")
    # Find frames where chessboard is detected in both Camera 1 and Camera cam_idx
    common_frames = set(frame_indices[0]).intersection(set(frame_indices[cam_idx]))
    if not common_frames:
        print(f"No common frames with detections for Wide cam and", camera_name[cam_idx])
        relative_extrinsics.append(None)
        continue

    # Collect corresponding object and image points
    objpoints_pair = []
    imgpoints1 = []
    imgpoints2 = []
    for video_idx, frame_count in common_frames:
        idx1 = frame_indices[0].index((video_idx, frame_count))
        idx2 = frame_indices[cam_idx].index((video_idx, frame_count))
        objpoints_pair.append(objp)
        imgpoints1.append(imgpoints[0][idx1])
        imgpoints2.append(imgpoints[cam_idx][idx2])

    # Perform stereo calibration
    if len(objpoints_pair) > 0:
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints_pair, imgpoints1, imgpoints2,
            intrinsics[0], dist_coeffs[0], intrinsics[cam_idx], dist_coeffs[cam_idx],
            (rois[0][2], rois[0][3]), flags=cv2.CALIB_FIX_INTRINSIC
        )
        if ret:
            extrinsic_matrix[cam_idx] = np.hstack((R, T))
            extrinsic_matrix[cam_idx] = np.vstack((extrinsic_matrix[cam_idx], [0, 0, 0, 1]))
            relative_extrinsics.append(extrinsic_matrix[cam_idx])
            print(f"Extrinsic Matrix (", camera_name[cam_idx], "relative to Wide cam):\n", extrinsic_matrix[cam_idx])
        else:
            print(f"Stereo calibration failed for Wide cam and", camera_name[cam_idx])
            relative_extrinsics.append(None)
    else:
        print(f"No valid common detections for Wide cam and", camera_name[cam_idx])
        relative_extrinsics.append(None)
# Save calibration data
for cam_idx in range(num_cameras):
    if cam_idx == 0 or relative_extrinsics[cam_idx - 1] is not None:
        with open(f'results/intrinsic_' + camera_name[cam_idx] + f'.json', 'w') as f:
            json.dump({'mtx': mtx.tolist(), 'dist': dist.tolist()}, f)
        with open(f'results/extrinsic_' + camera_name[cam_idx] + f'.json', 'w') as g:
            json.dump(extrinsic_matrix[cam_idx].tolist(), g)
        print(f"Calibration data saved for " + camera_name[cam_idx])
    else:
        print(f"No relative extrinsic matrix saved for " + camera_name[cam_idx])
