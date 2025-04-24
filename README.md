**Saving Babies With Machine Learning and Time Series Analysis**

**Overview**

This project investigates the use of machine learning and time series classification techniques to detect fetal movements based on wearable sensor data. The data was collected by a maternity hospital in Ireland and includes signals from piezoelectric sensors and maternal button presses used to indicate sensed fetal movements. Accurately detecting fetal movement is essential for prenatal care, offering critical insights into the health and wellbeing of the fetus.

**Problem Statement**

Time series data reflecting fetal movements presents an opportunity to better understand fetal health. With the help of machine learning, this project aims to:

Preprocess and structure a sensor dataset for machine learning.

Explore different strategies for generating labelled samples.

Evaluate various time series classification models.

Improve the precision of fetal movement detection to support better clinical outcomes.

**Dataset**

The dataset includes:

Wearable sensor recordings (piezoelectric signals).

Maternal button-press labels indicating perceived fetal movements.

Due to privacy restrictions, the dataset is not included in this repository. If authorised, you may access the dataset via secure institutional channels.

**Project Objectives**

Dataset exploration: Understand the structure of the data and perform necessary preprocessing.

Exploratory data analysis: Visualise patterns, frequencies, and durations of fetal movements.

Model development: Implement machine learning models to detect movements using labelled time series segments.

Evaluation: Validate models against ground truth labels and compare classifiers.

Interdisciplinary insight: Collaborate with experts from Computer Science and Biomedical Engineering for clinical relevance.

**Sample Generation Strategies**

Implemented in sample_generation.py:

Strategy 1: Pre/post click sampling using fixed non-overlapping windows.

Strategy 2: Non-overlapping fixed-length windows labelled based on the presence of clicks.

Strategy 3: Augmented positive samples by applying multiple shifts to windowed segments surrounding a click.

**Model Pipeline**

The main training and evaluation pipeline is implemented in main.py. It includes:

Sample generation and feature extraction.

Data splitting and standardisation.

**Training a classification pipeline using:**

QUANTTransformer()

StandardScaler()

LogisticRegression()

Performance evaluation using accuracy, precision, recall, and F1-score.

Dependencies

Install dependencies with:

pip install -r requirements.txt

Main packages:

numpy, pandas, scikit-learn

aeon for time series transformation and classification

matplotlib, seaborn for visualisation

File Structure

├── sample_generation.py     # Sample generation strategies
├── main.py                  # Main training and evaluation pipeline
├── utils.py                 # Helper functions for data loading and processing
├── config.py                # Configuration constants 
├── notebooks/               # Exploratory notebooks 
├── data/                    # Data folder (not included in repo)
├── requirements.txt         # Dependencies
└── README.md                # This file

**Acknowledgements**

This project is inspired by and builds upon:

Ghosh et al. (2024) - Multi-modal detection of fetal movements using a wearable monitor

Lai et al. (2016) - Fetal movements as a predictor of health

Middlehurst et al. (2023) - Evaluation of recent time series classification algorithms

**Disclaimer**

This repository does not include any patient-identifiable data. For access to the FeMo dataset, contact Colin Boyle and relevant authorities.

For questions or collaboration opportunities, please contact Treasa Murphy.
