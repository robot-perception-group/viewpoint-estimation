

# Viewpoint Estimation Pipeline

* **Species:** Optimized for Equids (Zebras)
* **Input Dimensions:** $224 \times 224$ bounding box crops
* **Downstream Integrations:** Embedded within `RAPID` and the `Behaviors Inference Framework` (add links)

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

* **`thesis`**: Contains the full pipeline and model training scripts. Use this branch to experiment, evaluate ablation configurations or train custom models.
* **`module`**: A lightweight, production-ready release containing only the core standalone estimator. Switch to this branch for direct, dependencies-minimized deployment and framework integration.