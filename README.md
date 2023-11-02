# AI Solution for Sustainable Apparel Product Classification

The objective of this project is to develop an AI solution using the Fashion MNIST dataset that aligns with our company's vision, with a primary focus on the identification and classification of sustainable apparel products.

## Dataset

* Dataset: Fashion MNIST 

* Fashion MNIST is a dataset containing grayscale images of clothing items categorized into 10 different classes.

## Task Description

### Data Analysis
* This includes exploration of the dataset to gain insights into the distribution of classes and the characteristics of the images.

* The script shows the class distribution and images of the classes after execution.
* The script includes a custom dataset model that takes annotations of the images and then predicts the outputs.

### Model Development
* The mode can design and train a machine learning or AI model capable of classifying apparel products based on the Fashion MNIST dataset.

* As the choice of model architecture and approach is flexible, the architecture is chosen randomly and it's a simple architecture.

## Evaluation Procedure

### Environment Setup
* Create a virtual environment and install the required dependencies listed in the requirements.txt file.

* Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the packages.

```bash
pip install -r requirements.txt
```

### Usage

```bash
python evaluate_model.py folder_containing_FashionMNIST_dataset/folder_containing_image_data
```

### Script Execution
* The solution includes a script named evaluate_model.py.

* This script accepts a command-line argument: the path to a folder containing the dataset for evaluation.

* Takes argument as a folder containing FashionMNIST Dataset or image files

* If there is no pre-trained model found in the project directory then training will start automatically after having the number of epochs as user input.

### Model Evaluation
* Within evaluate_model.py, the trained model will evaluate all data within the specified folder.

* The evaluation process produces relevant metrics, such as classification accuracy, to assess model performance.

### Output Generation
Post-evaluation, the script will generate an output.txt file that contains 
essential information, including:
* Summary of the model's architecture.
* Evaluation metric(s) obtained.
* Any additional insights or observations.
* Additional insights include missing or empty datasets, incompatible image size, Unmatched image types, etc.

### Error Handling 
* Common errors, such as missing or empty folders, will be handled gracefully. The script will provide informative error messages.

### Clean Exit
* After generating the output.txt file, the script should exit smoothly.
