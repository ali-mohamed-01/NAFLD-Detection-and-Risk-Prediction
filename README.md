# NAFLD Risk Prediction System

An AI-powered system for **Non-Alcoholic Fatty Liver Disease (NAFLD) risk prediction**, combining machine learning, data preprocessing, and Optical Character Recognition (OCR) to support data-driven risk assessment.

The project includes an end-to-end machine learning workflow, model evaluation, and a deployed API built with FastAPI and Hugging Face Spaces.

## Project Overview

Non-Alcoholic Fatty Liver Disease (NAFLD) is a condition characterized by fat accumulation in the liver that is not primarily caused by alcohol consumption.

This project aims to develop a machine learning solution that analyzes input features to predict the target class associated with NAFLD risk. It also integrates OCR technology to extract text from uploaded images and facilitate data input.

### Key Features

* **Machine Learning:** Classification using models such as Random Forest and Decision Tree.
* **Data Analysis:** Exploratory Data Analysis (EDA) and feature engineering.
* **Data Preprocessing:** Encoding categorical variables and scaling numerical features where appropriate.
* **Model Evaluation:** Performance assessment using precision, recall, F1-score, and accuracy.
* **OCR Integration:** Text extraction from images using Tesseract OCR and OpenCV.
* **API Development:** Model integration using FastAPI.
* **Cloud Deployment:** Deployment through Hugging Face Spaces.

## Technology Stack

| Category                | Technologies                                     |
| ----------------------- | ------------------------------------------------ |
| Programming Language    | Python                                           |
| Data Manipulation       | NumPy, Pandas                                    |
| Data Visualization      | Matplotlib, Seaborn                              |
| Machine Learning        | Scikit-learn                                     |
| Data Preprocessing      | OneHotEncoder, ColumnTransformer, StandardScaler |
| Model Persistence       | Joblib                                           |
| OCR                     | Tesseract OCR, Pytesseract, OpenCV               |
| API Development         | FastAPI                                          |
| Deployment              | Hugging Face Spaces                              |
| Development Environment | Jupyter Notebook                                 |

## Project Workflow

The machine learning pipeline follows these stages:

1. **Importing Dependencies** — Import the required Python libraries.
2. **Loading the Dataset** — Load the dataset and inspect its structure.
3. **Exploratory Data Analysis (EDA)** — Analyze distributions, relationships, missing values, and class balance.
4. **Feature Engineering** — Prepare relevant features for model training.
5. **Data Preprocessing and Transformation** — Encode categorical features and scale numerical data when required.
6. **Data Visualization** — Explore patterns and relationships through visualizations.
7. **Splitting the Data** — Separate the dataset into training and testing sets.
8. **Model Development** — Train and configure machine learning classification models.
9. **Model Evaluation** — Evaluate performance using classification metrics and a confusion matrix.
10. **Deployment and Integration** — Integrate the trained model into a FastAPI application and deploy it to Hugging Face Spaces.

## Model Performance

The model was evaluated on a test set containing **2,000 samples**.

### Classification Report

| Metric    | Class 0 | Class 1 |
| --------- | ------: | ------: |
| Precision |    0.95 |    0.74 |
| Recall    |    0.84 |    0.91 |
| F1-score  |    0.89 |    0.82 |
| Support   |   1,339 |     661 |

### Overall Metrics

| Metric                    | Score |
| ------------------------- | ----: |
| Accuracy                  |   86% |
| Macro Average Precision   |  0.84 |
| Macro Average Recall      |  0.88 |
| Macro Average F1-score    |  0.85 |
| Weighted Average F1-score |  0.87 |

### Performance Analysis

* **Accuracy (86%):** The model correctly classified approximately 86% of the test samples.
* **Class 1 Recall (91%):** The model identified 91% of the actual samples belonging to Class 1.
* **Class 1 Precision (74%):** Of the samples predicted as Class 1, 74% actually belonged to that class.
* **Weighted F1-score (87%):** The model achieved a weighted balance between precision and recall across both classes.

The relatively high recall for Class 1 may be useful when identifying potentially positive cases is an important objective. However, the precision score also indicates that some predictions may be false positives.

*Note: Class 0 and Class 1 should be mapped to their actual dataset labels. If Class 1 represents a positive NAFLD-risk outcome, the metrics above describe the model's performance for that outcome.*

## OCR and API Integration

The deployed application combines OCR processing with a machine learning prediction workflow.

### OCR Processing

The OCR component uses:

* **OpenCV:** Image loading and image preprocessing.
* **Pytesseract:** Text extraction using the Tesseract OCR engine.
* **Regular Expressions (re):** Pattern matching and text processing.

The extracted text can be processed to identify relevant information, depending on the application's implemented input format.

### FastAPI Backend

FastAPI provides the backend interface for integrating the trained model into an application.

Key components include:

* Handling uploaded files through `UploadFile` and `File`.
* Processing requests and validating input data.
* Loading serialized machine learning artifacts with Joblib.
* Integrating OCR and prediction logic.
* Returning structured prediction results through API endpoints.

### Deployment

The application is deployed using **Hugging Face Spaces**, making the system accessible through its hosted interface or API, depending on the deployment configuration.

* **Live Demo:** [Add your Hugging Face Space URL]
* **GitHub Repository:** [Add your GitHub repository URL]

## Project Structure

The following is a suggested repository structure. Adjust the paths to match your actual files.

```text
NAFLD-Risk-Prediction/
│
├── notebooks/
│   └── NAFLD_Model_Development.ipynb
│
├── models/
│   └── RandomForestClassifier_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Usage

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd NAFLD-Risk-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Install and configure the Tesseract OCR engine separately if it is not already available on your system.

### 4. Run the FastAPI Application

If your FastAPI application entry point is `app.py` and the FastAPI instance is named `app`, run:

```bash
uvicorn app:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The commands above assume your repository uses the suggested filenames and entry point.

## Future Improvements

* Evaluate additional classification algorithms and compare their performance.
* Perform cross-validation and hyperparameter tuning.
* Analyze the confusion matrix and class-specific error rates.
* Improve OCR accuracy through image preprocessing and text validation.
* Add clearer prediction explanations and confidence reporting where supported by the model.
* Validate performance on independent datasets.
* Improve API input validation, error handling, and automated testing.

## Limitations and Disclaimer

This project is intended for educational and research purposes. Its predictions should not be considered a medical diagnosis or a substitute for professional medical advice.

Model performance depends on the dataset, feature quality, preprocessing methods, and the representativeness of the evaluation samples. The reported test metrics alone do not establish clinical validity or real-world diagnostic performance.

## Author

**Ali Mohamed Ali**

AI Engineer | Machine Learning | Data Analysis

* GitHub: [ali-mohamed-01](https://github.com/ali-mohamed-01)
* Portfolio: [My Portfolio](https://ali-mohamed-01.github.io/My_Portfolio/)

---

*Developed as an applied machine learning project integrating data analysis, OCR, API development, and cloud deployment.*
