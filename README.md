# PneumoniaCXR: AI-Enabled Pneumonia Detection

## Introduction

This proof-of-concept project explores the use of advanced machine learning techniques, including convolutional neural networks and radiomics, to enhance the classification of pneumonia from chest X-ray images. By leveraging state-of-the-art AI methods, the system dynamically adapts its analysis to accurately distinguish between COVID-19, Non-COVID pneumonia, and normal cases, ensuring high diagnostic accuracy and supporting rapid medical response.

## Goals

- To develop and demonstrate a proof-of-concept for an AI system that accurately classifies different types of pneumonia from chest X-ray images.
- To assess the impact of various configurations of deep learning architectures and feature extraction methods on the classification accuracy.
- To optimize system performance through extensive testing and refinement of model parameters.

## Dataset

This project utilizes the COVID-QU-Ex Dataset, which contains over 30,000 chest X-ray images labeled as COVID-19 positive, non-COVID infections, and normal cases. Gold standard validation data is provided to benchmark the system's classification quality against expert radiological assessments.

### Source:
- **COVID-QU-Ex Dataset**: Anas M. Tahir, Muhammad E. H. Chowdhury, Yazan Qiblawey, Amith Khandakar, Tawsifur Rahman, Serkan Kiranyaz, Uzair Khurshid, Nabil Ibtehaz, Sakib Mahmud, and Maymouna Ezeddin, “COVID-QU-Ex .” Kaggle, 2021, [DOI: 10.34740/kaggle/dsv/3122958](https://doi.org/10.34740/kaggle/dsv/3122958). [Dataset available on Kaggle](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu).

## Methodology

1. **EDA and Data Preprocessing**:
   - Split data into train, validation, and test sets.
   - Resampled data to enable balanced classes.
   - Applied Z-scale normalization to images.
   - Adjusted cropping and alignment of images for uniformity.

2. **Feature Engineering**:
   - Extracted Histogram of Oriented Gradients (HOG), radiomics, and ResNet features.
   - Tuned HOG and radiomics extraction parameters for optimal performance.
   - Applied Principal Component Analysis (PCA) to reduce dimensionality and improve model efficiency.

3. **Modeling and Evaluation**:
   - Conducted hyperparameter tuning for models including SVM, logistic regression, random forest, and gradient boost.
   - Trained and tested models to evaluate their performance using metrics such as accuracy, precision, recall, and F1-score.

## Results

The results confirm that the PneumoniaCXR system effectively classifies chest X-ray images into COVID-19, non-COVID pneumonia, and normal cases with a high level of accuracy. The project achieved:
- 90% accuracy in model performance across different validation datasets, demonstrating good generalizability.
- Balanced importance among the different feature sets (HOG, radiomics, and ResNet), indicating that no single feature set dominated the predictive power.

## Usage

To utilize this project:
1. **Clone the repository** to your local machine.
2. **Install the required dependencies** using `pip install -r requirements.txt`.
3. **Run the Jupyter notebooks** in the `notebooks/` directory to conduct data preprocessing, perform feature extraction and selection, and build and evaluate machine learning models.

Note: Detailed instructions and code examples are provided in the Jupyter notebooks within the repository.

## Future Work

Further developments may include exploring additional imaging data sources for more robust classification, integrating more complex neural network architectures to enhance classification accuracy, and expanding the system customization options for different clinical settings.


## Future Work

While the current model demonstrates strong performance, future developments could include:
- Cross-validating generalizability across different CXR platforms and exploring CXR machine-specific models to tailor the system further for different imaging technologies and settings.
- Experimentation with additional model architectures and hybrid approaches to potentially enhance diagnostic accuracy.
- Extending the dataset to include more diverse demographic and geographic data to improve model robustness and applicability in varied clinical environments.

## Project Organization

## Project Organization

    ├── LICENSE
    ├── README.md                                                       <- The top-level README for developers using this project
    ├── data
    │   ├── raw                                                         <- The original, immutable raw dataset
    │   ├── preprocessed                                                <- Data after initial preprocessing (e.g., normalization, cropping)
    │   ├── features_extracted                                          <- Data after feature extraction and selection
    │   └── features_PCA                                                <- Data after dimensionality reduction (PCA applied)
    │
    ├── notebooks
    │   ├── 1.0-eda_data_preprocessing.ipynb                            <- Notebook for preprocessing images (loading, Z-scale normalization, cropping)
    │   ├── 2.0-feature_engineering.ipynb                               <- Notebook for HOG, radiomics and resnet feature extraction and dimensionality reduction
    │   └── 3.0-modelling                                               <- Notebook for model training and evaluation
    │
    ├── reports
    │   ├── figures                                                     <- Generated figures for the analysis
    │   ├── PneumoniaCXR-AIDetect_Presentation.pdf                      <- Generated figures for the analysis
    │   ├── figures                                                     <- Generated figures for the analysis
    │   ├── figures                                                     <- Generated figures for the analysis
    │   ├── figures                                                     <- Generated figures for the analysis
    │   └── final_report.ipynb                                          <- Final project report
    │
    └── models                                                          <- Trained and serialized models, model predictions, and diagnostics

Note: Due to file size limitations on GitHub, the dataset is hosted on Google Drive. Please download the data from the provided Google Drive link before running the notebooks.
