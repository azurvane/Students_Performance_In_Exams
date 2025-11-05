
# Predicting Student Exam Performance

## One-Sentence Description
Unleash the predictive power of **Linear Regression** to uncover the subtle factors driving student success across Math, Reading, and Writing scores in this exciting data analysis project!

## Features/Highlights
* **Custom Linear Regression Model:** Built from **scratch** using NumPy and Gradient Descent for a deep understanding of the core algorithm. *(See `LinearRegressionModel.py`)*
* **Score Grading System:** Raw scores are transformed into **binned 'grades' (0-4)** using `pd.cut()` for simplified classification and analysis.
* **Advanced Data Preprocessing:** Utilizes **Label Encoding** for binary features and **One-Hot Encoding** for multi-category features (`race/ethnicity`, `parental level of education`).
* **Visual Insights:** Explore **Seaborn countplots** and a **Correlation Heatmap** to visualize feature relationships and predict what truly matters.
* **Performance Metrics:** *Mean Squared Error (MSE)*, *Root Mean Squared Error (RMSE)*, *Mean Absolute Error (MAE)*, and the *R-squared ($R^2$) Score* calculated for each prediction model.

---

## Installation/Setup Guide 🛠️

### 4.1. Prerequisites
You need **Python 3.x** installed. The project relies on several common Python data science libraries.

### 4.2. Step-by-Step Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/azurvane/Students_Performance_In_Exams
    cd Students_Performance_In_Exams
    ```

2.  **Install dependencies:**
    The project uses the libraries listed in `requirements.txt`. Install them using pip:
    ```bash
    pip3 install -r requirements.txt
    ```

3.  **Ensure data file is present:**
    The notebook expects a file named `StudentsPerformance.csv` to be in the same directory.
    *(Note: This file is can be downloaded from the dataset source below.)*

---

## Usage Instructions 🚀

### 5.1. Getting Started:

The entire analysis, preprocessing, training, and evaluation is contained within the **`main.ipynb`** Jupyter Notebook.

1.  **Start Jupyter Notebook (or JupyterLab):**
    ```bash
    jupyter notebook
    # OR
    jupyter lab
    ```

2.  **Open `main.ipynb`:**
    Navigate to the file in your browser and open it.

3.  **Run all cells:**
    Execute the cells sequentially to perform the full data pipeline:
    * **Cells 1-7:** Data Loading and Initial Insights
    * **Cells 8-20:** Data Visualization (Countplots for binned scores)
    * **Cells 21-25:** Data Preprocessing (Binning Scores, Encoding Categorical Features, Correlation Map)
    * **Cells 26-40:** **Model Training and Evaluation** for Math, Reading, and Writing scores.

### 5.2. Custom Model Snippet: 

To get a quick fix of how the custom `LinearRegression` class works, you can use this snippet (also demonstrated in `TestModel.ipynb`):

```python
import numpy as np
from LinearRegressionModel import LinearRegression

# 1. Initialize a model with 4 features
model = LinearRegression(n_features=4, random_state=1) 

# 2. Prepare dummy training data 
X_train = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
y_train = np.array([10.0, 30.0])

# 3. Train the model – watch the loss plummet!
loss_history = model.train(X_train, y_train, iterations=5000, learning_rate=0.01)

# 4. Make a new prediction
X_new = np.array([1.5, 2.5, 3.5, 4.5])
prediction = model.predict(X_new)

print(f"Prediction for X_new: {prediction[0]:.2f}")
# Output will be a close approximation to the actual linear relationship!
````

-----

## Configuration ⚙️

| File/Variable | Description | Default/Example |
| :--- | :--- | :--- |
| `main.ipynb` | **Core analysis and training script.** Adjust parameters here. | N/A |
| `iterations` (Cell 2) | Number of training epochs for Gradient Descent. | `100000` |
| `learning_rate` (Cell 2) | Step size for gradient descent optimization. | `0.01` |
| `bins`, `labels` (Cell 8) | Defines the boundaries and labels for score binning (0-4). | `[-1, 59.99, 69.99, 79.99, 89.99, 100]`, `[0, 1, 2, 3, 4]` |

-----

## Credits/Acknowledgments 🙏

The dataset used for this analysis is the **Student Performance in Exams** data, generously provided via Kaggle.

  * **Dataset Link:** [https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

<!-- end list -->
