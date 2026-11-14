# Tibia X-ray Image Segmentation Using U-Net

## 1. Project Overview

This project was completed as part of the Machine Vision and Neural Networks course. The aim was to develop a deep learning model that segments the tibia region in X-ray images.

I implemented a U-Net convolutional neural network using PyTorch. The model was trained using X-ray images and their corresponding annotated masks. After training, I evaluated its performance on validation images and generated predictions for a separate test set.

## 2. Project Objectives

The main objectives of this project were:

- Prepare and preprocess X-ray images and segmentation masks.
- Understand and implement the U-Net architecture.
- Train a neural network for binary image segmentation.
- Evaluate segmentation performance using the Dice coefficient.
- Generate and visualize segmentation predictions.
- Experiment with different prediction thresholds.

## 3. Dataset and Preprocessing

The dataset used for this project contained:

- 150 X-ray images.
- 100 annotated segmentation masks.
- Original images and processed masks prepared at 224 × 224 pixels.

The dataset was divided into three groups:

| Dataset | Number of images |
|---|---:|
| Training | 80 |
| Validation | 20 |
| Testing | 50 |
| **Total X-ray images** | **150** |

The 100 annotated images were used for training and validation. The remaining 50 X-ray images were reserved for testing.

The preprocessing steps included:

- Preparing the X-ray images as grayscale images.
- Processing the segmentation masks as binary masks.
- Resizing images and masks to 224 × 224 pixels.
- Normalizing image and mask pixel values.
- Creating CSV files to organize the training, validation and test datasets.

**Dataset availability:** The original dataset is not included in this public repository. It must be obtained separately, subject to its original licence and usage conditions.

## 4. U-Net Model Architecture

I implemented a U-Net model using PyTorch.

The architecture contains:

- An encoder for extracting image features.
- A bottleneck for deeper feature representation.
- A decoder for reconstructing the segmentation output.
- Skip connections to preserve spatial information.
- A final convolutional layer that produces one-channel segmentation output.

The model receives grayscale X-ray images with one input channel and produces a segmentation prediction with one output channel.

## 5. Model Training

The model was trained using the following configuration:

| Parameter | Value |
|---|---|
| Deep learning framework | PyTorch |
| Architecture | U-Net |
| Input channels | 1 |
| Output channels | 1 |
| Image size | 224 × 224 |
| Epochs | 20 |
| Batch size | 4 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Loss function | Combined BCE and Dice Loss |

During training, the model was evaluated using the validation dataset. The best model checkpoint was saved based on validation performance.

The best saved checkpoint came from epoch 19.

## 6. Model Evaluation

After training, I evaluated the saved model using the 20 validation images.

The evaluation produced the following result:

| Metric | Result |
|---|---:|
| Number of validation images | 20 |
| Average Dice score | 0.7208 |
| Average Dice percentage | 72.08% |
| Selected prediction threshold | 0.6 |
| Best checkpoint epoch | 19 |

### Prediction Threshold Comparison

I tested six different prediction thresholds to see how they affected the average validation Dice score.

| Threshold | Average Dice |
|---|---:|
| 0.3 | 70.77% |
| 0.4 | 71.08% |
| 0.5 | 71.49% |
| 0.6 | **72.08%** |
| 0.7 | 63.55% |
| 0.8 | 21.62% |

Among the six tested thresholds, 0.6 achieved the highest average validation Dice score. Therefore, I used this threshold when generating the final predictions.

## 7. Segmentation Results

The project includes visual comparisons of:

1. Original X-ray images.
2. Ground-truth segmentation masks.
3. Predicted segmentation masks.

Five validation comparison images are saved in:

`outputs/plots/`

The model also generated:

- 20 validation predictions, saved in `outputs/predictions/validation/`.
- 50 test predictions, saved in `outputs/predictions/test/`.

The predicted masks capture parts of the target region, but they are not identical to the ground-truth masks. Some differences remain in the segmentation boundaries.

The test images do not have corresponding annotated masks. Therefore, predictions were generated for all 50 test images, but a quantitative test Dice score could not be calculated.

## 8. Technologies and Libraries

The project was developed using:

- Python
- PyTorch
- NumPy
- Pandas
- Pillow (PIL)
- Matplotlib

## 9. Project Structure

```text
Image-Segmentation/
│
├── models/
│   └── unet.py
│
├── train.py
├── evaluate.py
├── dataset.py
├── dice_score.py
├── process_masks.py
├── create_csv.py
├── split_csv.py
├── update_csv.py
├── check_masks.py
├── check_thresholds.py
├── create_chart.py
│
├── requirements.txt
├── csv_sample_counts.png
├── .gitignore
└── README.md
```

The original dataset, trained model checkpoint and generated prediction images are excluded from the public repository.

## 10. How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/puja0711/tibia-xray-image-segmentation.git
```

### Open the Project Folder

```bash
cd tibia-xray-image-segmentation
```

### Install the Required Libraries

```bash
pip install -r requirements.txt
```

### Prepare the Dataset

Obtain the dataset separately and prepare the image, mask and CSV folders according to the paths expected by the project.

The dataset must be prepared separately before training. The evaluation script requires a trained checkpoint generated by `train.py`.

### Train the Model

```bash
python train.py
```

### Evaluate the Model

```bash
python evaluate.py
```

The evaluation script loads the best saved checkpoint, evaluates the validation images and generates predictions for the test images.

## 11. Project Limitations

Some limitations of this project are:

- The number of annotated training images was limited to 80.
- The predicted masks do not perfectly match the ground-truth boundaries.
- Model performance may vary between different X-ray images.
- The 50 test images do not have ground-truth masks, so quantitative test evaluation was not possible.
- Further experimentation with model architecture, training settings and preprocessing could improve the results.

## 12. Conclusion

In this project, I developed and trained a U-Net model for tibia X-ray image segmentation using PyTorch.

The model achieved an average validation Dice score of 72.08% using a prediction threshold of 0.6. It also generated segmentation predictions for 50 test images.

Through this assignment, I gained practical experience in image preprocessing, binary segmentation, U-Net implementation, neural network training, model evaluation and visualization.