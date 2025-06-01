# AI/ML Internship - Task 4: Classification with Logistic Regression - Medical Diagnosis Prediction

## Objective
The main objective of this task was to build a binary classifier using Logistic Regression to predict medical diagnosis, focusing on understanding key classification concepts and evaluation metrics.

## Dataset
The dataset used for this task is the [data.csv](data.csv) dataset, which contains features related to cell characteristics and a 'diagnosis' (Benign or Malignant).

## Tools and Libraries Used
* **Python**
* **Pandas:** For data loading and manipulation.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For machine learning model implementation (Logistic Regression, train-test split, StandardScaler) and evaluation metrics.
* **Matplotlib:** For creating static visualizations (ROC Curve, Confusion Matrix, Sigmoid Function, Precision-Recall Curve).
* **Seaborn:** For enhanced statistical graphics (used for heatmap visualization of Confusion Matrix).

## Logistic Regression Steps Performed:

### 1. Choose a Binary Classification Dataset
* Loaded the `data.csv` dataset.
* Performed initial essential preprocessing by dropping the 'id' column (identifier) and the 'Unnamed: 32' column (empty).
* Encoded the 'diagnosis' target variable ('M' for Malignant to 1, 'B' for Benign to 0) to prepare it for binary classification.
* **Outcome:** The dataset was prepared with numerical features and a binary numerical target.

### 2. Train/Test Split and Standardize Features
* Separated features (X) from the target (y).
* Split the dataset into training (70%) and testing (30%) sets using `train_test_split`, ensuring stratification to maintain class proportions.
* Standardized all numerical features using `StandardScaler` on the training data and then transformed both training and testing sets. This step is crucial for Logistic Regression to ensure features are on a similar scale and prevent dominance by features with larger values.
* **Outcome:** Data was correctly partitioned and scaled, ready for model training.

### 3. Fit a Logistic Regression Model
* Initialized and trained a `LogisticRegression` model from `sklearn.linear_model` on the standardized training data (`X_train_scaled`, `y_train`).
* **Outcome:** The model learned the logistic relationships between the scaled features and the binary diagnosis.

### 4. Evaluate with Confusion Matrix, Precision, Recall, ROC-AUC
* Made predictions on the standardized test set (`X_test_scaled`).
* Calculated and displayed key classification evaluation metrics based on your execution results:
    * **Accuracy:** 0.9708
    * **Precision:** 0.9836
    * **Recall:** 0.9375
    * **F1-score:** 0.9600
* Generated and presented the **Confusion Matrix**:
    ```
    [[106   1]
     [  4  60]]
    ```
    This indicates 1 False Positive and 4 False Negatives, demonstrating strong performance with a high ability to correctly identify positive cases while minimizing incorrect positive predictions.
* Calculated the ROC AUC score (0.9975 from your output), indicating an excellent ability to distinguish between classes.
* Plotted the **Confusion Matrix** and the **Receiver Operating Characteristic (ROC) Curve** (including AUC score) for visual evaluation.
* **Outcome:** Provided a comprehensive quantitative and visual assessment of the model's classification performance.

### 5. Tune Threshold and Explain Sigmoid Function
* **Sigmoid Function Explanation:** Provided a detailed explanation of how the sigmoid (logistic) function maps the linear combination of features into a probability between 0 and 1, which is central to Logistic Regression. A plot of the sigmoid function was included.
* **Threshold Tuning Demonstration:** Showed how the default classification threshold of 0.5 can be adjusted to balance precision and recall based on specific problem requirements. An example with a new threshold (e.g., 0.3) was used to illustrate its impact on metrics and the confusion matrix.
* Plotted the **Precision-Recall Curve**, which is particularly useful for selecting the optimal threshold for imbalanced datasets or when the costs of False Positives and False Negatives differ significantly.
* **Outcome:** Enhanced understanding of how Logistic Regression probabilities are derived and how the classification decision can be fine-tuned.

## Visualizations
The repository includes the following generated plots:
* `confusion_matrix.png`: A heatmap visualization of the model's confusion matrix.
* `roc_curve_task4.png`: The Receiver Operating Characteristic (ROC) curve with AUC score.
* `sigmoid_function.png`: A plot illustrating the sigmoid function.
* `precision_recall_curve.png`: A plot showing the trade-off between precision and recall at different thresholds.

## Conclusion
This task successfully demonstrated the implementation of a Logistic Regression binary classifier, including data preparation, model training, comprehensive evaluation using various metrics, and an exploration of threshold tuning and the underlying sigmoid function.
