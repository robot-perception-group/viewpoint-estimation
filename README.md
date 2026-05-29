

# Viewpoint Estimation Pipeline
This repository contains the code and models to accompany my thesis on "Viewpoint Estimation for Animal Re-Identification on Edge Devices" at the Max Planck Institute for Intelligent Systems
* **Species:** Optimized for Equids (Zebras)
* **Input Dimensions:** $224 \times 224$ bounding box crops
* **Downstream Integrations:** Embedded within `RAPID` and the [`Behaviors Inference Framework`](https://github.com/robot-perception-group/Animal-Behaviour-Inference-Framework)
---
[Demo Video](https://www.youtube.com/watch?v=7CFVS-LiWfc)


https://github.com/user-attachments/assets/3ff355d4-a38e-477b-acef-3ea29a74ad63


---
```mermaid

graph LR
    %% Style Definitions
    classDef data fill:#f4f7fa,stroke:#b0c4de,stroke-width:1.5px,color:#2c3e50,font-size:13px;
    classDef module fill:#fff5eb,stroke:#ffa07a,stroke-width:2px,stroke-dasharray: 0,color:#2c3e50,font-weight:bold,font-size:13px;

    %% Nodes
    A["Input Image<br>(224×224 Crop)"]:::data
    B["Custom YOLO<br>Pose Model"]:::module
    C["17 Keypoints with<br>confidence scores"]:::data
    D["Ablated<br>Feature Set"]:::data
    E["Viewpoint<br>MLP"]:::module
    F["Regressed Angle<br>(φ)"]:::data
    G["Keypoint skeleton"]:::data

    %% Connections
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    C -. Optional  .-> G

```






## Repository Architecture

This repository is organized into two separate branches:

* **`master`**: Contains the full pipeline and model training scripts from my thesis. Use this branch to experiment, evaluate ablation configurations or train custom models.
* **`module`**: A lightweight, production-ready release containing only the core standalone estimator. Switch to this branch for direct, dependencies-minimized deployment and framework integration.
