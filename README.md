# Facial Expression Recognition using PyTorch

This project focuses on building a deep learning model to classify facial expressions into one of seven categories: **angry**, **disgust**, **fear**, **happy**, **neutral**, **sad**, and **surprise**. The model is built using PyTorch and leverages a pre-trained EfficientNet-B0 architecture for transfer learning.

## Project Overview
Facial Expression Recognition (FER) is a computer vision task that involves classifying human emotions based on facial images. This project uses a deep learning model trained on the [Facial Expression Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) to achieve this task. The model is trained using PyTorch and achieves high accuracy in classifying emotions.

### Key Features:
- Uses **EfficientNet-B0** as the backbone architecture.
- Implements data augmentation techniques like random horizontal flipping and rotation.
- Evaluates the model on a validation set and saves the best weights.
- Includes an inference script to visualize predictions on sample images.

---

## Dataset
The dataset used in this project is the **Facial Expression Dataset**, which contains images categorized into seven emotions:
1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

The dataset is split into training and validation sets. You can download the dataset from [here](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).

---

## Installation
To set up the project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Baneet2s/facial-expression-recognition.git
   cd facial-expression-recognition
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.7+ installed. Then, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can install the dependencies manually:
   ```bash
   pip install torch torchvision timm albumentations opencv-python tqdm matplotlib numpy
   ```

3. **Download the Dataset:**
   Clone the dataset repository:
   ```bash
   git clone https://github.com/parth1620/Facial-Expression-Dataset.git
   ```

---

## Usage

### Training the Model
To train the model, run the following Python script:
```python
python train.py
```

### Inference
To perform inference on a sample image from the validation set, use the following code:
```python
python inference.py
```

### Hyperparameters
You can customize the training process by modifying the following hyperparameters in the script:
- `LR`: Learning rate (default: `0.001`)
- `BATCH_SIZE`: Batch size (default: `32`)
- `EPOCHS`: Number of epochs (default: `15`)
- `DEVICE`: Training device (`'cuda'` for GPU or `'cpu'` for CPU)

---

## Results
After training, the model achieves the following performance:
- **Training Accuracy:** ~90%
- **Validation Accuracy:** ~85%

### Sample Inference
Below is an example of the model's prediction on a sample image:

![Sample Inference](sample_inference.png)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
