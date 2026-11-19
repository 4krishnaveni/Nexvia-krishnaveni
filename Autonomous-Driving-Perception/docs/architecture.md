# Autonomous Driving Perception Stack — Architecture

```mermaid
flowchart TD
    A["Input: KITTI-style Dataset / CARLA Camera Frame"] --> B["Image Preprocessing"]
    B --> C["YOLO Object Detection"]
    C --> D["ByteTrack Object Tracking"]

    B --> E["Classical Lane Detection"]
    B --> F["SegFormer Road Segmentation"]
    E --> G["Lane and Road Fusion"]
    F --> G

    B --> H["MiDaS Monocular Depth Estimation"]

    D --> I["Perception Fusion"]
    G --> I
    H --> I

    I --> J["Road Status Estimation"]
    J --> K["Temporal Smoothing"]
    K --> L["Annotated Output and CSV Results"]

    M["CARLA Simulator"] --> N["RGB Camera Capture"]
    N --> B
