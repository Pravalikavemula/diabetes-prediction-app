# Diabetes Prediction Project 🩺

A machine learning project that predicts whether a person has diabetes based on medical attributes using Python and Scikit-Learn.

## 📂 Files
- `diabetes_prediction.py`: Data analysis, model training, evaluation, and saving.
- `app.py`: Streamlit web app to predict diabetes.
- `diabetes_model.pkl`: Saved Random Forest model.
- `requirements.txt`: Dependencies to install.
- `README.md`: Documentation.

## 📊 Dataset
Pima Indian Diabetes Dataset from [UCI ML Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## 🛠️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/diabetes_prediction_project.git
cd diabetes_prediction_project

#testing
#Run your app.py again and test extreme inputs like:
#Glucose = 200
#BMI = 40
#Insulin = 500
#You should get "Diabetic" now.