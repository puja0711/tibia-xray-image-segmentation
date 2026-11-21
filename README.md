# Tibia X-ray Image Segmentation Using LeXT U-Net

## 1. Project Overview

This project was completed as part of the Machine Vision and Neural Networks course. The aim was to develop a deep learning model that segments the tibia region in X-ray images.

I developed a binary image segmentation model using PyTorch. The project uses a modified U-Net implementation developed by Aleksei Tiulpin from the University of Oulu. The model was trained using X-ray images and their corresponding annotated masks.

After training, I evaluated the model using validation images and generated segmentation predictions for a separate test set.

## 2. Project Objectives

The main objectives of this project were:

- Prepare and preprocess X-ray images and segmentation masks.
- Understand and use the U-Net architecture.
- Train a neural network for binary image segmentation.
- Evaluate segmentation performance using the Dice coefficient.
- Generate and visualize segmentation predictions.
- Improve model training and evaluation.

## 3. Dataset and Preprocessing

The dataset used for this project contained:

- 150 X-ray images.
- 100 annotated segmentation masks.
- Images and processed masks resized to 224 × 224 pixels.

The dataset was divided into three groups:

| Dataset | Number of images |
|---|---:|
| Training | 80 |
| Validation | 20 |
| Testing | 50 |
| **Total X-ray images** | **150** |

The 100 annotated images were used for training and validation. The remaining 50 X-ray images were reserved for testing.

The preprocessing steps included:

- Converting X-ray images to grayscale.
- Processing segmentation masks as binary masks.
- Resizing images and masks to 224 × 224 pixels.
- Normalizing pixel values.
- Creating CSV files to organize the training, validation and test datasets.

**Dataset availability:** The original dataset is not included in this public repository. It must be obtained separately, subject to its original licence and usage conditions.

## 4. U-Net Model Architecture

This project uses a modified U-Net implementation developed by Aleksei Tiulpin, University of Oulu (2017).

The implementation supports different network depths, widths and activation functions.

The model configuration used in this project includes:

- Basic width: 24
- Depth: 6
- Center depth: 2
- Input channels: 1
- Output channels: 1
- Convolution type: Regular
- Activation function: ReLU

The architecture contains an encoder, a bottleneck, a decoder and skip connections. It receives grayscale X-ray images and produces one-channel segmentation predictions.

The model implementation is available in `models/unet_lext.py`.

## 5. Model Training

The model was trained using the following configuration:

| Parameter | Value |
|---|---|
| Deep learning framework | PyTorch |
| Architecture | LeXT U-Net |
| Input channels | 1 |
| Output channels | 1 |
| Image size | 224 × 224 |
| Epochs | 20 |
| Batch size | 4 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Loss function | Combined BCE and Dice Loss |
| Random seed | 42 |

The training process uses a combination of Binary Cross-Entropy loss and Dice loss.

Random seeds were set to help make the training process reproducible.

The best model checkpoint was saved based on validation Dice performance.

The best checkpoint was obtained at epoch 19.

## 6. Model Evaluation

After training, I evaluated the best saved checkpoint using the 20 validation images.

The evaluation produced the following result:

| Metric | Result |
|---|---:|
| Number of validation images | 20 |
| Average Dice score | 0.9637 |
| Average Dice percentage | **96.37%** |
| Prediction threshold | 0.5 |
| Best checkpoint epoch | 19 |

The Dice score was calculated separately for each validation image, and the final result represents the average across all 20 images.

## 7. Segmentation Results

The project generates visual comparisons containing:

1. Original X-ray images.
2. Ground-truth segmentation masks.
3. Predicted segmentation masks.

The validation comparison images are saved in:

`outputs/plots/`

The model also generated:

- 20 validation predictions in `outputs/predictions/validation/`.
- 50 test predictions in `outputs/predictions/test/`.

The predictions show the segmented tibia regions. Some differences between predicted and ground-truth boundaries may still remain.

The 50 test images do not have corresponding annotated masks. Therefore, predictions were generated for all 50 test images, but a quantitative test Dice score could not be calculated.

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
│   ├── unet.py
│   └── unet_lext.py
│
├── data/
│   └── CSVs/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── checkpoints/
├── outputs/
│   ├── plots/
│   └── predictions/
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
├── LICENSE
├── .gitignore
└── README.md
```

The original dataset, trained model checkpoints and generated prediction images are excluded from the public repository.

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

The dataset must be prepared before training.

### Train the Model

```bash
python train.py
```

The best checkpoint is saved as:

`checkpoints/best_model_lext.pth`

### Evaluate the Model

```bash
python evaluate.py
```

The evaluation script loads the saved checkpoint, evaluates the validation images and generates predictions for the test images.

## 11. Project Limitations

Some limitations of this project are:

- The number of annotated training images was limited to 80.
- The predicted masks may not perfectly match the ground-truth boundaries.
- Model performance may vary between different X-ray images.
- The 50 test images do not have ground-truth masks, so quantitative test evaluation was not possible.
- Further experimentation with model architecture, training settings and preprocessing could improve the results.

## 12. License and Attribution

This project uses a modified U-Net implementation developed by Aleksei Tiulpin, University of Oulu (2017).

The original copyright notice is preserved in the model implementation.

The implementation is distributed under the MIT License. The original `LICENSE` file is included in this repository.

## 13. Conclusion

In this project, I developed and trained a U-Net-based model for tibia X-ray image segmentation using PyTorch.

The improved LeXT U-Net model achieved an average validation Dice score of 96.37% across 20 validation images. It also generated segmentation predictions for 50 test images.

Through this assignment, I gained practical experience in image preprocessing, binary segmentation, U-Net architectures, neural network training, model evaluation and visualization.