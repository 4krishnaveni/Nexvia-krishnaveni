# Autonomous Driving Perception Stack
## Technical Project Report

### 1. Introduction
This project implements a computer vision-based perception pipeline for autonomous driving scenarios. It combines object detection, object tracking, lane detection, road segmentation, monocular depth estimation, and road-status analysis.

### 2. Objectives
- Detect vehicles, pedestrians, and other road objects.
- Track detected objects across video frames.
- Identify lane boundaries and estimate road regions.
- Estimate approximate depth using monocular vision.
- Analyze road-position status and apply temporal smoothing.
- Demonstrate camera capture and offline inference using CARLA.

### 3. Technologies Used
- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- ByteTrack
- SegFormer
- MiDaS
- CARLA Simulator 0.9.16
- Pandas and NumPy

### 4. System Implementation
The pipeline contains the following modules:

1. YOLO object detection.
2. ByteTrack multi-object tracking.
3. Classical lane detection using image processing.
4. SegFormer-based road segmentation.
5. Lane and road-region fusion.
6. MiDaS monocular depth estimation.
7. Road-status estimation and temporal smoothing.
8. Annotated video and CSV result generation.

### 5. Dataset
A KITTI-style dataset containing 4,290 label files and 22,917 annotated objects was used.

The dataset was divided by video:
- Training: Video1–Video13 (2,976 frames)
- Validation: Video14–Video16 (834 frames)
- Testing: Video17–Video18 (773 frames)

### 6. Experimental Results

#### YOLO baseline
- Precision: 0.308
- Recall: 0.217
- mAP@50: 0.215
- mAP@50–95: 0.109

#### Lane detection
- Classical lane detection success: 364/773 frames (47.09%)
- Road-mask and lane fusion success: 390/773 frames (50.45%)
- Improvement: 3.36 percentage points

These figures represent lane-boundary fit success, not ground-truth lane-detection accuracy.

#### Temporal smoothing
- Raw road-status transitions: 171
- Smoothed transitions: 76
- Reduction: 55.56%

This measures transition reduction, not road-status classification accuracy.

#### CARLA demonstration
- Simulator: CARLA 0.9.16
- Map: Town10HD_Opt
- Camera image resolution: 640 × 360
- YOLO offline image inference: 1 detection
- Offline demonstration video: Generated and playback verified

The offline demonstration repeats a single captured image. Continuous live CARLA-to-YOLO processing was attempted but was unstable because of simulator and graphics-device errors.

### 7. Limitations
- The YOLO baseline has limited precision and recall.
- Lane detection is sensitive to lighting and road appearance.
- Monocular depth is approximate and has not been validated against LiDAR ground truth.
- Road-status labels have not been evaluated against manually verified ground truth.
- Continuous CARLA camera processing was not reliably completed.
- No real vehicle control or autonomous braking was implemented.
- TensorRT acceleration was not tested because the development laptop does not have an NVIDIA GPU.

### 8. Conclusion
The project demonstrates an integrated experimental perception pipeline combining object detection, tracking, lane and road analysis, approximate depth estimation, and temporal smoothing. The experiments identify useful implementation results as well as limitations that require further validation.

### 9. Safety Disclaimer
This is an educational research prototype. It is not validated or intended for real-world vehicle control, collision avoidance, or autonomous driving deployment.
