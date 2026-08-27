# Major Project: ML in Manufacturing

Three independent case studies applying classical ML, computer vision, and transfer learning to common manufacturing-line problems: predicting equipment failure types, segmenting/classifying products on a line, and flagging defective products for quality control.

Each subfolder is a self-contained Jupyter notebook + dataset/artifacts — there's no shared pipeline between them.

| Project | Problem | Approach | Result |
|---|---|---|---|
| [`Failure Type Prediction`](#1-failure-type-prediction) | Predict *which* type of machine failure is occurring from sensor readings | Classical ML (LinearSVC, KNN, Random Forest) on tabular sensor data | Tuned Random Forest: **90% accuracy**, 0.90 precision/recall/F1 (weighted avg) |
| [`Product Segmentation`](#2-product-segmentation) | Detect and group products (bottles) moving on a conveyor belt | YOLOv3 object detection → MobileNetV2 feature extraction → K-Means clustering | Real-time video pipeline, image and webcam modes |
| [`Quality Assurance`](#3-quality-assurance) | Flag damaged vs. undamaged products for QC | Transfer learning (MobileNetV2) binary classifier, exported to quantized TFLite for on-device inference | **89.5% accuracy** (macro F1 ≈ 0.89) on held-out test set |

## Repository Structure

```
Failure Type Prediction/
  Failure Type Prediction.ipynb      Full notebook: EDA, modeling, evaluation
  predictive_maintenance.csv         Cleaned/working dataset (10,000 rows)
  predictive_maintenance_dataset.csv Raw source dataset

Product Segmentation/
  Product Segmentation.ipynb         Full notebook: detection + clustering pipeline
  yolov3.cfg                         YOLOv3 network config (weights not included — see Setup)
  coco.names                         COCO class labels used by YOLOv3
  Test Images/                       Sample conveyor-belt images for testing
  final/                             Output image(s) from a pipeline run

Quality Assurance/
  Quality Assurance.ipynb            Full notebook: data prep, training, TFLite export, live demo
  data.csv                           Labeled image regions (damaged/undamaged) — VGG Image Annotator export
  cascade1.xml                       Haar cascade used for live product localization
  tflite_model_quant.tflite          Quantized, exported classifier for on-device inference
```

## 1. Failure Type Prediction

**Goal:** given live sensor readings from a machine (temperatures, rotational speed, torque, tool wear), predict which of six outcomes is occurring — `No Failure`, `Heat Dissipation Failure`, `Power Failure`, `Overstrain Failure`, `Tool Wear Failure`, or `Random Failures` — rather than just a binary "will it fail" flag.

- **Dataset:** predictive-maintenance style dataset, 10,000 rows. Heavily imbalanced (96.5% `No Failure`); the notebook downsamples the majority class to improve learning signal on the minority failure types.
- **Features:** air/process temperature, rotational speed, torque, tool wear, and product type.
- **Approach:** three models trained and compared — LinearSVC, KNN, and Random Forest — with 5-fold cross-validation, followed by `RandomizedSearchCV` hyperparameter tuning on the Random Forest.
- **Result:** the tuned Random Forest reached **90% test accuracy** with 0.90 weighted precision/recall/F1. Performance is strong on the majority classes and on `Overstrain Failure`/`Heat Dissipation Failure`, and weaker on the rarest class (`Random Failures`, 18 examples total) — expected given how few examples exist to learn from.

## 2. Product Segmentation

**Goal:** identify and group individual products (bottles) as they move along a manufacturing belt, so a downstream mechanism can sort or separate them.

- **Approach:** two techniques are combined —
  1. **YOLOv3** object detection locates individual bottles in each frame.
  2. Each detected bottle crop is passed through a pretrained **MobileNetV2** (feature extraction only, no fine-tuning) and the resulting feature vectors are grouped with **K-Means clustering** to segment bottles by visual type.
- **Modes:** the notebook supports both a static-image pipeline (`realtime_img`) and a live webcam/video pipeline (`realtime_vid`), the latter with CUDA backend/target configured for GPU inference.
- **Note:** `yolov3.weights` (the pretrained YOLOv3 weights file, ~236 MB) is not committed to this repo due to size — see [Setup](#setup) below for how to get it.

## 3. Quality Assurance

**Goal:** classify a product image as **damaged** or **undamaged** for automated visual QC.

- **Dataset:** `data.csv` — image region annotations (bounding boxes + damage label) exported from the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/), covering 1,859 labeled product images.
- **Approach:** transfer learning on a pretrained **MobileNetV2** (via TensorFlow Hub), fine-tuned on the damaged/undamaged labels with an early-stopping callback and TensorBoard logging. Trained with an 80/20 train-val/test split (and a further 75/25 split of the training portion into train/validation).
- **Result:** **89.5% test accuracy**, with class-wise F1 of 0.88 (undamaged) and 0.90 (damaged) — reasonably balanced given the dataset isn't heavily skewed toward one class.
- **Deployment path:** the trained Keras model is converted to a **quantized TFLite model** (`tflite_model_quant.tflite`) for lightweight on-device inference, and paired with a **Haar cascade** (`cascade1.xml`) for real-time product localization from a live camera feed before classification.

## Setup

Each notebook was originally run locally (Windows) and contains **hardcoded local file paths** (e.g. `D:/My Work/Mechatronics/...`) for data, model, and output directories — update these to match your own environment before running.

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow tensorflow-hub opencv-python seaborn dataframe_image
```

- **Failure Type Prediction** needs no extra assets — `predictive_maintenance.csv` is included.
- **Product Segmentation** needs `yolov3.weights` in the project folder (not committed — download from the [official YOLOv3 release](https://pjreddie.com/media/files/yolov3.weights) or the [Darknet repo](https://github.com/pjreddie/darknet)), alongside the included `yolov3.cfg` and `coco.names`.
- **Quality Assurance** needs a webcam for the live-detection cells; the pretrained `tflite_model_quant.tflite` and `cascade1.xml` are already included, so the training cells can be skipped if you just want to run inference.

## Paper

This repo is the code accompanying:

> R. Mahapatra and K. Gaurav, "Real Time Quality Assessment in a Production Line: Machine Learning Approach," 2023 (MoSICom). [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10458752)

The paper covers the same three areas as this repo — failure type prediction (LinearSVC/KNN/Random Forest), product segmentation (K-Means), and quality assurance (MobileNetV2) — with the methodology and results write-up.

## Citation

If you use code from this repository, please cite the paper above.

## License

No license file is currently included — treat the code as all-rights-reserved unless you hear otherwise from the author.
