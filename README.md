# **Fabric Defect Detection Project**

### **Overview**
This project aims to automate defect detection in fabrics for the textile industry, using advanced deep learning techniques to improve quality control processes. Detectable defects include holes, starting marks, missing wefts, and others. By developing a machine learning model that accurately detects and classifies defects in real-time, we aim to replace traditional, labor-intensive inspection methods with an efficient, consistent, and scalable solution.

### **Problem Statement**
Manual fabric defect detection is inefficient, costly, and can strain workers' vision over prolonged periods. This project focuses on studying common fabric defects, creating a dataset of defect images, and developing a model to automatically detect and classify them, thereby minimizing human error and enhancing inspection accuracy.

### **Proposed Solution**
An automated detection system will be developed to identify and classify fabric defects from captured images. By leveraging machine learning models like CNN, YOLO, and RT-DETR, our solution will enable real-time inspection, reduce manual labor, and ensure consistent quality standards. The ultimate goal is to integrate this solution with textile manufacturing machinery for continuous, real-time monitoring.

---

## **Table of Contents**
- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Setup](#setup)
- [Usage](#usage)
- [Documentation Structure](#documentation-structure)
- [Project Management](#project-management)
- [Contributors](#contributors)

---

## **Project Goals**
1. **Data Collection**: Capture and label images from provided fabric samples, focusing on defects such as holes, starting marks, and missing wefts.
2. **Model Training and Testing**: Train and evaluate deep learning models for accurate defect detection and classification.
3. **Real-time Detection**: Enable the detection model to process images in real time when integrated with rolling machinery.
4. **Industrial Integration**: Implement the detection system into Machine Matic’s machinery to assist with quality control in textile manufacturing.

---

## **Dataset**
We are using high-resolution images captured by a VCG-X camera, set up in collaboration with Machine Matic. The dataset comprises images with and without defects across five different fabric samples, each labeled for specific defect types.

- **Current Size**: 1500 images with initial focus on two defect types.
- **Collection Method**: Images are captured with a fabric roller setup.
- **Annotations**: Defects are manually annotated with bounding boxes and labels for model training.

---

## **Models Used**
The project leverages multiple models for optimal defect detection:

1. **CNN (Convolutional Neural Network)** - For initial classification tasks.
2. **YOLO (You Only Look Once)** - For real-time object detection in fabric images.
3. **RT-DETR (Real-Time Detection Transformer)** - For robust and real-time defect localization on rolling fabric.

Each model is fine-tuned and evaluated based on accuracy, detection speed, and suitability for real-time deployment.

---

## **Setup**

### **Requirements**
- Python 3.x
- Libraries:
  - `opencv-python`
  - `numpy`
  - `matplotlib`
  - `torch` (for PyTorch-based models)
  - `scikit-image`
  - `tensorflow` (for alternative deep learning models)
- **Install Dependencies**:
  ```bash
  pip install opencv-python numpy matplotlib torch scikit-image tensorflow
