# Breast Cancer Analysis

This repository contains a machine learning project focused on breast cancer analysis using the CBIS-DDSM: Breast Cancer Image Dataset. The project's focus is on the pre-processing of mammogram images, feature extraction, and the application of various machine learning algorithms to classify breast cancer.

## Features

TODO

## Dataset

The dataset used in this project is the [CBIS-DDSM: Breast Cancer Image Dataset](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset). It contains mammogram images with annotations for breast cancer classification. The dataset includes:

- 1: Negative
- 2: Benign finding
- 3: Probably Benign
- 4: Suspicious finding (4a, 4b, 4c)
- 5: Highly suggestive of malignancy
- 6: Malignant (biopsy proven)

The classes are re-mapped into the csv files to 7-14, where:

- 7: Negative
- 8: Benign finding
- 9: Probably Benign
- 10: Suspicious finding (4a)
- 11: Suspicious finding (4b)
- 12: Suspicious finding (4c)
- 13: Highly suggestive of malignancy
- 14: Malignant (biopsy proven)

Then they are re-mapped to 0-7 for model usage, where:

- 0: Negative
- 1: Benign finding
- 2: Probably Benign
- 3: Suspicious finding (4a)
- 4: Suspicious finding (4b)
- 5: Suspicious finding (4c)
- 6: Highly suggestive of malignancy
- 7: Malignant (biopsy proven)

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

5. Download the dataset from [INbreast Release 1.0](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset) and place it the project's root directory.

6. Run the `data.ipynb` notebook to preprocess the data and generate the augmented dataset.

7. Run the `preprocessing-and-model.ipynb` notebook to train and evaluate the machine learning model through the various possible preprocessing techniques.

> ![NOTE]
> The recommended python version for the notebook is 3.11.8, as some libraries may not be compatible with earlier versions. This is also the most recent version compatible with TensorFlow GPU usage.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
