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

## Results
After training, the model achieves the Validation Accuracy of ~64%

### Sample Inference
Below are some examples of the model's prediction on sample images:
- ![Image](https://github.com/Baneet2s/Facial-Expression-Recognition-with-PyTorch/blob/main/Samples/Sample%201.png)
- ![Image](https://github.com/Baneet2s/Facial-Expression-Recognition-with-PyTorch/blob/main/Samples/Sample%202.png)
- ![Image](https://github.com/Baneet2s/Facial-Expression-Recognition-with-PyTorch/blob/main/Samples/Sample%203.png)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
