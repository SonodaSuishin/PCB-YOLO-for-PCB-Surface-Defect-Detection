# PCB-YOLO: Lightweight PCB Surface Defect Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-6.0-green.svg)](https://github.com/ultralytics/yolov5)

A lightweight and efficient object detection model for PCB surface defect detection, balancing accuracy and computational efficiency for industrial edge deployment. This repository contains the complete training pipeline and an online detection system prototype.

---

## 📌 Overview

Printed Circuit Board (PCB) surface defect detection is a critical task in industrial quality control. This project proposes **PCB-YOLO**, an improved YOLOv5-based model that achieves a balance between detection accuracy and computational efficiency, making it suitable for deployment on resource-constrained edge devices.

**Key Contributions:**
- **C3FFT Module**: Incorporates Fast Fourier Transform (FFT) for global feature extraction with minimal parameter overhead
- **C3A2 Module**: Integrates Region Attention with Flash Attention optimization for enhanced feature focusing
- **CGAFusion Module**: Content-Guided Attention fusion for adaptive multi-scale feature aggregation
- Maintains only **2.45M parameters** while achieving competitive accuracy (mAP@0.5: 0.970)

<img width="707" height="288" alt="image" src="https://github.com/user-attachments/assets/cb1985f7-3d3d-4c01-81de-62fe189666b0" />

---

## 🏗️ Model Architecture

PCB-YOLO is built upon YOLOv5s with three key architectural improvements:

### 1. C3FFT Module (Shallow Layer Enhancement)
Placed at the shallow layers of the backbone, this module leverages Fast Fourier Transform to capture global texture features efficiently. It uses a residual fusion structure to preserve spatial details while incorporating frequency-domain information.

<img width="523" height="272" alt="image" src="https://github.com/user-attachments/assets/319e50f3-8f63-47a3-a2a9-88c80f090bf1" />


### 2. C3A2 Module (Deep Layer Enhancement)
Applied to deep backbone layers and neck layers, this module integrates Region Attention with Flash Attention optimization, enabling the model to focus on discriminative local regions crucial for small defect detection.

<img width="657" height="166" alt="image" src="https://github.com/user-attachments/assets/a3d865bb-09bb-4356-b013-3b3247918856" />


### 3. CGAFusion Module (Feature Fusion Enhancement)
Replaces the standard concatenation operation in the neck network with a Content-Guided Attention mechanism, dynamically weighting features from different scales for optimal fusion.

<img width="708" height="314" alt="image" src="https://github.com/user-attachments/assets/87231668-31b3-4720-ad3c-97a6cb1eaafb" />


### Overall Architecture

<img width="552" height="564" alt="image" src="https://github.com/user-attachments/assets/caa0db11-7d68-4dc9-8689-b512f5b9fc97" />


---

## 📊 Dataset

The model is trained and evaluated on the **DeepPCB** dataset, which contains:
- 1,500 high-resolution PCB images (640×640)
- 6 types of defects: open, short, mousebite, spur, pinhole, spurious_copper
- 22,000+ annotated defect instances

Training was conducted on Google Colab with the following configuration:
- **Batch Size**: 16
- **Epochs**: 300
- **Optimizer**: SGD with momentum
- **Learning Rate**: 0.01 (cosine annealing)

---

## 📈 Results

### Baseline Comparison

| Model | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs | FPS | Score S |
|-------|---------|--------------|------------|--------|-----|---------|
| **PCB-YOLO (Ours)** | **0.970** | 0.752 | **2.45** | **7.1** | 123.5 | 0.5805 |
| YOLOv5n | 0.986 | 0.794 | 2.50 | 7.1 | 476.2 | 0.5901 |
| YOLOv10n | 0.961 | 0.746 | 2.27 | 6.5 | 212.8 | 0.5738 |
| YOLOv11n | 0.901 | 0.573 | 2.58 | 6.3 | 384.6 | 0.5388 |
| YOLOv12s | 0.929 | 0.626 | 9.23 | 21.2 | 125.0 | 0.5325 |
| YOLOv8s | 0.978 | 0.738 | 11.1 | 28.4 | 243.9 | 0.5535 |
| Faster R-CNN | 0.937 | 0.717 | 41.7 | 7.5 | 40.0 | 0.4845 |

### Pareto Front Analysis

<img width="583" height="373" alt="image" src="https://github.com/user-attachments/assets/85ff1ab7-369c-411f-a5a6-31d8ef9c8626" />


As shown above, **PCB-YOLO** lies on the Pareto frontier, achieving an optimal balance between accuracy (mAP@0.5) and lightweight cost (Params + GFLOPs). While YOLOv5n offers slightly higher accuracy and YOLOv10n offers slightly lower cost, PCB-YOLO provides the best trade-off for practical industrial deployment where both factors matter.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Google Colab account (recommended for training)

### Training

The entire training pipeline is implemented and best weight file is given. Simply run all cells in sequence.

1. **Environment Setup**: Install dependencies and configure Colab environment
2. **Data Preparation**: Download and prepare DeepPCB dataset
3. **Model Definition**: Build PCB-YOLO with custom modules (C3FFT, C3A2, CGAFusion)
4. **Training**: Train with specified hyperparameters
5. **Evaluation**: Evaluate on test set and generate metrics

### Online Detection System

After training, the model can be deployed using the provided detection interface. Upload PCB images to get real-time defect detection results with bounding boxes and confidence scores.

<img width="694" height="421" alt="image" src="https://github.com/user-attachments/assets/df9a7358-4a2f-49b4-9ec2-0ce7d0f3a30c" />
<img width="575" height="455" alt="image" src="https://github.com/user-attachments/assets/39c2256c-cd78-4373-b11d-bb0e736e041b" />

---

## 📁 Repository Structure

Since the project is implemented as a Jupyter Notebook, the code is organized sequentially within `pcb_yolo_system.ipynb`:
```code
├── pcb_yolo_system.ipynb # Complete training pipeline
├── final.pt # best weight to load
├── loss.py # loss function to use
├── yolov5_C3FFTFixed_C3A2_CGAFusion.yaml # model structure configuration file
├── hyp_PCB.yaml # hyp-parameters used for data augmentation
└── README.md # This file
```

** Note **:This project extends Ultralytics YOLOv5. To reproduce the training you should clone the YOLO project from Ultralytics.The clone commands are given in the .ipynb file.

---

## 🔍 Key Technologies

| Technology | Purpose |
|------------|---------|
| **Fast Fourier Transform (FFT)** | Global feature extraction with O(N log N) complexity |
| **Region Attention** | Efficient spatial attention via region-level aggregation |
| **Flash Attention** | Memory-efficient attention computation for GPUs |
| **Content-Guided Attention (CGA)** | Adaptive multi-scale feature fusion |
| **YOLOv5** | Baseline architecture for lightweight detection |

---

## 📝 Citation

If you find this work helpful for your research, please cite:

```bibtex
@article{pcb-yolo,
  title={PCB-YOLO: Lightweight PCB Surface Defect Detection with FFT and Attention Mechanisms},
  author={[Your Name]},
  year={2026}
}
```
---

### 📧 Contact
For questions,collaboration opportunities or suggestion, feel free to reach out:

Email: wyannchenn@163.com

GitHub: SonodaSuishin
