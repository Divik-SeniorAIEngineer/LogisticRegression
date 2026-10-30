# Logistic Regression & Classification Metrics

Exploring **Logistic Regression** from the ground up starting from the Perceptron trick and Sigmoid activation to Batch Gradient Descent, Polynomial Logistic Regression, Softmax Multiclass Classification, and comprehensive Binary/Multiclass Evaluation Metrics (Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC).

---

## 📁 Notebooks Overview

### 1. Foundations & Optimization (From Scratch vs. Scikit-Learn)

| Notebook                                                                   | Description                                                                                                                                                                                                                  | Key Concepts                                                                                            | Dataset                                           |
| :------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ | :------------------------------------------------ |
| [`perceptron_logistic.ipynb`](./perceptron_logistic.ipynb)                 | Implements the **Perceptron Trick** from scratch using a unit step activation function, animates the decision boundary across training epochs, and compares the resulting boundary with Scikit-Learn's `LogisticRegression`. | Step function, Perceptron weight update rule, Decision boundary animation (`FuncAnimation`)             | Synthetic (`make_classification`, `class_sep=10`) |
| [`perceptron_logistic_sigmoid.ipynb`](./perceptron_logistic_sigmoid.ipynb) | Extends the Perceptron algorithm by replacing the step function with the **Sigmoid** activation function and compares three decision boundaries: Step Perceptron, Sigmoid Perceptron, and Scikit-Learn `LogisticRegression`. | Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$, probabilistic weight updates vs. hard step updates | Synthetic (`make_classification`, `class_sep=30`) |
| [`logistic_gradient_descent.ipynb`](./logistic_gradient_descent.ipynb)     | Implements **Batch Gradient Descent** for Logistic Regression from scratch and compares the learned coefficients, intercept, and decision boundary against `LogisticRegression(solver='sag')`.                               | Cross-entropy gradient update, Batch Gradient Descent (`gd`), Sigmoid activation                        | Synthetic (`make_classification`, `class_sep=20`) |

---

### 2. Non-Linear & Multiclass Extensions

| Notebook                                                                               | Description                                                                                                                                                                                                                                                  | Key Concepts                                                                                                      | Dataset                           |
| :------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :-------------------------------- |
| [`polynomial_regression.ipynb`](./polynomial_regression.ipynb)                         | Demonstrates how to capture non-linear decision boundaries using **Polynomial Features** with `LogisticRegression` across degrees $1, 2, 3, 4, 5, 6, 7,$ and $25$, evaluating 10-fold cross-validation accuracy and visualizing overfitting at high degrees. | `PolynomialFeatures`, `cross_val_score`, non-linear decision boundary contour plots, underfitting vs. overfitting | `../data/ushape.csv`              |
| [`softmax_multiclass_classification.ipynb`](./softmax_multiclass_classification.ipynb) | Applies **Softmax Regression** (Multinomial Logistic Regression with `solver='lbfgs'`) to classify Iris species using sepal and petal length, predicts class probabilities on new queries, and plots decision regions.                                       | Softmax / Multinomial Logistic Regression, `LabelEncoder`, `plot_decision_regions` (`mlxtend`)                    | `../data/iris.csv`                |
| [`multiclass_classification_mnist.ipynb`](./multiclass_classification_mnist.ipynb)     | Performs 10-class image classification on the **Fashion-MNIST** dataset comparing `LogisticRegression` and `DecisionTreeClassifier` using weighted precision, recall, F1-score, and full classification reports.                                             | High-dimensional multiclass classification, weighted metrics (`average='weighted'`), `classification_report`      | `../data/fashion-mnist_train.csv` |

---

### 3. Classification Evaluation Metrics

| Notebook                                                                       | Description                                                                                                                                                                                                                                            | Key Concepts                                                                                                                                                                       | Dataset                |
| :----------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------- |
| [`binary_classification_metrics.ipynb`](./binary_classification_metrics.ipynb) | Evaluates and compares `LogisticRegression` and `DecisionTreeClassifier` on binary heart disease prediction using Accuracy, Confusion Matrix, Precision, Recall, and F1-Score (both binary and per-class).                                             | `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `average=None`                                                                                                  | `../data/heart.csv`    |
| [`multi_classification_metrics.ipynb`](./multi_classification_metrics.ipynb)   | Evaluates and compares `LogisticRegression` and `DecisionTreeClassifier` on the 3-class Iris dataset using Confusion Matrices and per-class Precision and Recall scores.                                                                               | Multiclass confusion matrix, per-class precision & recall (`average=None`)                                                                                                         | `../data/iris.csv`     |
| [`ROC-AOC.ipynb`](./ROC-AOC.ipynb)                                             | Computes and visualizes interactive **ROC Curves** and **AUC** scores using Plotly for `LogisticRegression` and `SVC`, annotates probability thresholds, and finds the optimal classification threshold maximizing $TPR - FPR$ (Youden's J statistic). | `roc_curve`, `roc_auc_score`, False Positive Rate (FPR) vs. True Positive Rate (TPR), Optimal Threshold ($\arg\max(TPR - FPR)$), Model comparison (`LogisticRegression` vs. `SVC`) | `../data/diabetes.csv` |

---

## 🧠 Core Mathematical Concepts Covered

1. **Sigmoid Activation**:

   $$
   \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^T \mathbf{x} + b
   $$

2. **Batch Gradient Descent Update**:

   $$
   \mathbf{w} \leftarrow \mathbf{w} + \eta \cdot \frac{1}{N} \mathbf{X}^T (\mathbf{y} - \hat{\mathbf{y}})
   $$

3. **2D Linear Decision Boundary Extraction**:

   $$
   w_1 x_1 + w_2 x_2 + b = 0 \implies x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}
   $$

   where slope $m = -\frac{w_1}{w_2}$ and intercept $c = -\frac{b}{w_2}$.

4. **Evaluation Metrics**:
   - $\text{Precision} = \frac{TP}{TP + FP}$
   - $\text{Recall (TPR)} = \frac{TP}{TP + FN}$
   - $\text{FPR} = \frac{FP}{FP + TN}$
   - $\text{F1-Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
   - $\text{Optimal Threshold} = \text{thresholds}[\arg\max(\text{TPR} - \text{FPR})]$

---

## 🛠️ Requirements & Setup

Ensure the following Python packages are installed:

```bash
pip install numpy pandas scikit-learn matplotlib mlxtend plotly jupyter
```

### Data Directory Structure

Several notebooks expect datasets located in `../data/` relative to this folder:

- `../data/diabetes.csv`
- `../data/heart.csv`
- `../data/iris.csv`
- `../data/fashion-mnist_train.csv`
- `../data/ushape.csv`
