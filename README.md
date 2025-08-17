# ML WWarehouse Optimize Project

This project builds and evaluates machine learning models to **predict food demand** using data from a meal delivery company.  
It covers data preprocessing, exploratory data analysis (EDA), model training, evaluation, and inference.

---


## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/ml-sales-project.git
   cd ml-sales-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv SalesMLenv
   source SalesMLenv/bin/activate   # Linux / Mac
   SalesMLenv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   - For **inference only**:
     ```bash
     pip install -r requirements.txt
     ```
   - For **training & EDA**:
     ```bash
     pip install -r requirements-dev.txt
     ```

---

## Workflow

### 1. Data Preprocessing
Run:
```bash
python ./src/preprocess.py
```
- Merges `train.csv`, `fulfilment_center_info.csv`, and `meal_info.csv`
- Splits dataset into **train/val/test sets**

### 2. Model Training
Run:
```bash
python ./src/train_model.py
```
- Trains **Linear Regression** and **Random Forest Regressor**
- Evaluates with **MAE** and **R²**
- Saves models into `Models/` as `.pkl` files

### 3. Prediction / Inference
Run:
```bash
python ./src/predict.py
```
- Loads the best saved model (`random_forest_model.pkl`)
- Runs predictions on test data

---

## Example Results

```
[XGBoost]   MAE = 136.7904 | R² = 0.7864
[RandomForestReg]    MAE = 78.1390  | R² = 0.7864
```

Random Forest performs significantly better and is chosen for inference.

---

## Tech Stack
- Python 3.10+
- pandas, numpy
- scikit-learn
- joblib
- matplotlib, seaborn
- Jupyter Notebook

---

## Future Improvements
- Hyperparameter tuning with **GridSearchCV / Optuna**
- Feature engineering (e.g., lag features, demand seasonality)
- Deploy as **REST API** (FastAPI/Flask)
- Model monitoring in production

---

## Author
Developed by **Hoang Phucs**  
Contact: hoangphuc0826@gmail.com
