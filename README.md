# Breast Cancer Analysis

This repository presents a comprehensive study on breast cancer analysis using the INbreast mammography dataset. The project emphasizes advanced image preprocessing, data augmentation, and the application of deep learning models—including custom CNNs and state-of-the-art architectures—to improve the classification of breast cancer in mammogram images. All experiments, results, and the full study PDF are included for reproducibility and further research.

## Features

The project includes the following features:

- **Case Study Paper**: [A detailed paper discussing the case study, methodologies, and results of the breast cancer analysis.](main.pdf)
- **Data Preprocessing**: Implementation of various preprocessing techniques such as denoising, binarization, low-pass filtering, and morphological operations to enhance image quality and improve classification accuracy.
- **Model Training**: Training of multiple machine learning models, including custom convolutional neural networks (CNNs) and transfer learning models based on ResNet50 and DenseNet121, to classify breast cancer images.

## Study PDF

The repository includes the full study as a PDF document. You can find and read the detailed paper in the `article/main.pdf` file.

## Dataset

The dataset used in this project is the [CBIS-DDSM: Breast Cancer Image Dataset](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset). It contains mammogram images with annotations for breast cancer classification. The dataset includes:

- 1: Negative
- 2: Benign finding
- 3: Probably Benign
- 4: Suspicious finding (4a, 4b, 4c)
- 5: Highly suggestive of malignancy
- 6: Malignant (biopsy proven)

## Installation

The project is run in a Jupyter Notebook environment. To set up the environment, follow these steps:

1. Clone and navigate to the repository:
   ```bash
    git clone TODO
    cd breast-cancer-analysis
   ```

2. Install the required packages:
   ```bash
    pip install -r requirements.txt
    ```

3. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open the notebook files in your browser.

5. Download the dataset from [INbreast Release 1.0](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset) and place it the project's `/data` directory.

6. Run the `data.ipynb` notebook to preprocess the data and generate the augmented dataset.

7. Run the `preprocessing-and-model.ipynb` notebook to train and evaluate the machine learning model through the various possible preprocessing techniques.

8. The results will be generated and saved in the `data/results` directory.

> [!NOTE]
> The recommended python version for the notebook is 3.12.3

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
