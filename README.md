# Autonomous Driving Perception Stack

## 1. Project Overview

An experimental computer-vision-based autonomous driving perception system that integrates object detection, multi-object tracking, road-area segmentation, lane detection, depth estimation, and temporal road-status analysis.

The system processes driving-scene images and videos to identify road users, track detected objects, estimate relative depth, and analyze whether detected objects overlap with the segmented road area.

**Purpose:** Educational and research demonstration only. This system is not validated or intended for real-world vehicle control.

## 2. Key Features

- **Object Detection:** YOLOv8-based detection of vehicles and other road users.
- **Multi-Object Tracking:** ByteTrack-based object tracking and track IDs.
- **Lane Detection:** Classical computer-vision lane detection using HLS, Canny edge detection, ROI masking, and Hough transforms.
- **Road Segmentation:** SegFormer-based road-area segmentation.
- **Depth Estimation:** MiDaS-based relative depth estimation with approximate pinhole distance calibration.
- **Road-Status Analysis:** Estimates whether detected objects overlap with the segmented road region.
- **Temporal Smoothing:** Uses a rolling five-observation history to reduce road-status fluctuations.
- **CARLA Integration:** CARLA 0.9.16 virtual camera setup and captured simulation frames.
- **Offline YOLO Demonstration:** Runs YOLO inference on a captured CARLA image and generates an annotated MP4 video.
- **Evaluation:** CSV logs, visualizations, and processed videos.

## 3. Technology Stack

- Python 3.12
- PyTorch (CPU)
- Ultralytics YOLOv8
- ByteTrack
- OpenCV
- Hugging Face Transformers
- SegFormer
- MiDaS
- NumPy
- Pandas
- CARLA Simulator 0.9.16

## 4. System Architecture

The perception pipeline follows these stages:

1. Input driving-scene images or video.
2. YOLOv8 detects vehicles and other road users.
3. ByteTrack associates detections across video frames.
4. SegFormer generates a road-area segmentation mask.
5. MiDaS estimates relative depth.
6. Object bounding-box road coverage is calculated.
7. Road status is assigned as ON_ROAD, OFF_ROAD, or UNKNOWN.
8. Temporal smoothing reduces frame-to-frame status fluctuations.
9. Annotated videos and CSV evaluation logs are generated.

### CARLA Demonstration Workflow

1. Launch CARLA Simulator 0.9.16.
2. Connect to the simulator through the CARLA Python API.
3. Access the Town10HD_Opt simulation environment.
4. Attach an RGB virtual camera to a simulated vehicle.
5. Capture a simulation image.
6. Run YOLO inference on the captured image.
7. Generate an annotated offline demonstration video.

**Note:** Continuous live CARLA-to-YOLO video processing was attempted but could not be stabilized because of simulator and Direct3D graphics-device errors. The verified demonstration uses a captured CARLA frame.

## 5. Dataset

The project uses the provided KITTI-style driving dataset containing Video1 through Video18, with image frames and corresponding labels.

Dataset split:

- Training: Video1–Video13
- Validation: Video14–Video16
- Testing: Video17–Video18

Dataset analysis:

- 4,290 label files
- 22,917 annotated bounding boxes
- 105 empty label files

## 6. Experimental Results

### YOLOv8 Baseline

Five-epoch CPU baseline evaluation:

| Metric | Result |
|---|---:|
| Precision | 0.308 |
| Recall | 0.217 |
| mAP@50 | 0.2154 |
| mAP@50–95 | 0.1094 |

These results represent an experimental baseline, not production-level detection performance.

### Lane Detection

Full test-set comparison:

| Method | Successful line fits |
|---|---:|
| Classical lane detection | 364 / 773 (47.09%) |
| Lane fusion | 390 / 773 (50.45%) |

Success means both lane-line fits were detected. It does not represent ground-truth lane accuracy.

### Object Tracking

ByteTrack processed 340 frames from Video17 and generated tracked object IDs.

### Temporal Road-Status Smoothing

Video16 evaluation:

| Metric | Result |
|---|---:|
| Tracked detections | 666 |
| Unique tracks | 40 |
| Raw status transitions | 171 |
| Smoothed status transitions | 76 |
| Transitions reduced | 95 |
| Reduction | 55.56% |

The reduction demonstrates improved temporal consistency, not validated road-classification accuracy.

### CARLA and YOLO Demonstration

| Component | Result |
|---|---|
| CARLA version | 0.9.16 |
| Simulation map | Town10HD_Opt |
| Python API connection | Successful |
| Captured image resolution | 640 × 360 |
| YOLO detections on captured frame | 1 |
| Offline video generation | Successful |
| Offline video playback | Verified |
| Offline video duration | 5 seconds |

The single-frame detection is a demonstration result, not a measure of detection accuracy. The generated video repeats the same annotated frame.

## 7. Hardware and Execution Environment

Development environment:

- Operating System: Windows
- Processor: Intel Core Ultra 5 125H
- Graphics: Intel Arc integrated graphics
- RAM: 16 GB
- Python: 3.12
- PyTorch: CPU-based execution

CUDA acceleration was not available in this environment.

CARLA rendering required reduced resolution and low-quality settings because of graphics-device stability issues.

## 8. Important Project Files

### Source Code

- `src/pipeline.py` – Integrated perception pipeline.
- `src/lane_classical.py` – Classical lane detection.
- `src/lane_segmentation.py` – Segmentation-based road analysis.
- `src/carla_camera.py` – CARLA virtual camera capture.
- `src/carla_yolo_video.py` – Experimental live CARLA and YOLO video script.
- `src/carla_yolo_offline.py` – Offline YOLO demonstration video generation.

### CARLA Outputs

- `results/carla_camera/carla_camera_frame.png` – Captured CARLA camera image.
- `results/carla_yolo/carla_yolo_offline_demo.mp4` – Verified offline YOLO demonstration video.

### Other Outputs

Important outputs are stored in the `results/` directory:

- `bytetrack_video17.mp4`
- `road_status_smoothed_video16.mp4`
- `pipeline_log.csv`
- `temporal_smoothing_evaluation.csv`
- `road_status_video16.csv`
- `bottom_center_video16.csv`
- `road_coverage_video16.csv`
- Lane detection and segmentation visualizations
- Depth calibration configuration

## 9. Project Limitations

- The YOLOv8 baseline has limited detection performance.
- Lane-line detection is not consistently reliable.
- SegFormer road segmentation is not lane-marking detection.
- MiDaS depth is relative; pinhole calibration is approximate.
- Road-status thresholds are experimental and require ground-truth validation.
- Temporal smoothing improves consistency but cannot correct an incorrect underlying classification.
- CARLA live video processing was unstable on the available integrated graphics hardware.
- The offline CARLA demonstration uses one captured frame repeated in a video.
- No real-time vehicle-control or braking functionality is enabled.

## 10. Safety Disclaimer

This project is strictly for educational and research purposes. It is not a certified autonomous-driving system and must not be used to control a real vehicle or make safety-critical driving decisions.

## 11. Conclusion

The project demonstrates an integrated experimental perception pipeline combining object detection, tracking, road segmentation, depth estimation, temporal analysis, and CARLA simulation-based camera capture.

The measured 55.56% reduction in road-status transitions demonstrates the benefit of temporal smoothing, while the CARLA offline YOLO demonstration illustrates inference on a simulated driving-scene image.

The evaluation results also identify areas requiring further improvement, particularly detection performance, lane detection reliability, depth validation, and stable real-time simulator integration.