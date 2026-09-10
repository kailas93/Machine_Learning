# Supervised Learning — Complete Cheatsheet
### Regression & Classification: Definitions, Mathematics, Parameters, Metrics & Code

---

# 1. Introduction to Supervised Learning

**Definition:** Supervised learning is a branch of machine learning where a model learns a mapping function `f` from input features `X` to a known output/label `y`, using a labeled training dataset `{(x1,y1), (x2,y2), ..., (xm,ym)}`. Once trained, the model predicts `y` for new, unseen `x`.

```
ŷ = f(x)
```

| Term | Meaning |
|---|---|
| `x` (or `X`) | Input feature(s) / independent variable(s) |
| `y` | True/actual output (label / target) |
| `ŷ` (y-hat) | Predicted output |
| `m` | Number of training examples |
| `n` | Number of features |
| `w` (or `θ`) | Weight(s) / coefficient(s) learned by the model |
| `c` (or `b`) | Bias / intercept term |

### 1.1 Two Categories

| Type | Output `y` | Goal | Examples |
|---|---|---|---|
| **Regression** | Continuous real number | Predict a quantity | House price, temperature, salary |
| **Classification** | Discrete category/class | Predict a label | Spam/Not spam, disease Y/N, digit 0–9 |

### 1.2 General ML Workflow

1. Collect & clean data
2. Split into Train / Test (and often Validation) sets
3. Select a model
4. **Fit / Train:** learn parameters `w, c` by minimizing a cost function
5. **Predict:** apply the learned function to new data
6. **Evaluate:** measure performance using metrics (Section 10 & 11)
7. **Tune:** adjust hyperparameters (e.g., via GridSearchCV) and repeat

### 1.3 Key Definitions Used Throughout

- **Model / Hypothesis function:** the mathematical formula used to make predictions, denoted `f(x)`, `h(x)`, or `ŷ`.
- **Parameters:** values learned from data during training (weights `w`, bias `c`).
- **Hyperparameters:** values set *before* training (e.g., learning rate, number of trees, `k`) that control the learning process.
- **Cost / Loss function `J`:** measures how wrong the model's predictions are; training = minimizing `J`.
- **Overfitting:** model memorizes training data (low train error, high test error).
- **Underfitting:** model is too simple to capture patterns (high error on both train & test).
- **Regularization:** technique to penalize large weights and reduce overfitting.
- **Bias-Variance Tradeoff:** balance between a model being too simple (high bias) vs too sensitive to training data (high variance).

---

# PART A — REGRESSION ALGORITHMS

---

## A1. Linear Regression

**Definition:** Linear Regression models the relationship between one or more independent variables and a continuous dependent variable by fitting the **best straight line** (or hyperplane) through the data, minimizing the sum of squared errors.

### Simple Linear Regression (1 feature)

Uses the familiar straight-line equation:

```
y = m*x + c
```

| Symbol | Meaning |
|---|---|
| `y` | predicted/dependent variable |
| `x` | independent variable (feature) |
| `m` | slope of the line (how much y changes per unit x) — also written `w` or `θ1` |
| `c` | y-intercept (value of y when x = 0) — also written `b` or `θ0` |

### Multiple Linear Regression (n features)

Generalizes to vector form:

```
ŷ = w1*x1 + w2*x2 + ... + wn*xn + c   =  wᵀx + c
```

Where `w = [w1, w2, ..., wn]` is the weight vector and `x = [x1, x2, ..., xn]` is the feature vector. (This is the same as `θᵀx` notation used in many textbooks, with `θ0 = c`.)

**Cost Function — Mean Squared Error (MSE):**

```
J(w,c) = (1/m) * Σ (ŷ_i - y_i)²     for i = 1 to m
```

Some texts use `1/2m` so the derivative simplifies:
```
J(w,c) = (1/2m) * Σ (ŷ_i - y_i)²
```

**Finding best `m` and `c` (Least Squares, single feature):**

```
m = Σ[(x_i - x̄)(y_i - ȳ)] / Σ[(x_i - x̄)²]
c = ȳ - m*x̄
```
where `x̄` and `ȳ` are the means of x and y.

**Closed-form solution for multiple features (Normal Equation):**

```
w = (XᵀX)⁻¹ Xᵀ y
```

**Assumptions of Linear Regression:**
1. Linearity — relationship between X and y is linear
2. Independence of errors
3. Homoscedasticity — constant variance of residuals
4. Normal distribution of residuals
5. No/low multicollinearity among features

**Key Parameters (`LinearRegression`):**
| Parameter | Meaning |
|---|---|
| `fit_intercept` | Whether to compute intercept `c` (default True) |
| `n_jobs` | CPU cores to use for computation |
| `copy_X` | Whether to copy X before fitting (avoid overwriting) |
| `positive` | Force coefficients to be positive if True |

**Sample Code:**
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Slope (m):", model.coef_)
print("Intercept (c):", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

---

## A2. Polynomial Regression

**Definition:** Polynomial Regression extends linear regression by adding powers of the input feature (x², x³, …) so the model can fit **curved (non-linear)** relationships, while remaining linear in its parameters.

**Math (degree n):**

```
y = w1*x + w2*x² + w3*x³ + ... + wn*xⁿ + c
```

This is still of the form `ŷ = wᵀΦ(x) + c`, where `Φ(x) = [x, x², x³, ..., xⁿ]` is the transformed feature vector — hence it's solved using the same linear regression machinery after feature transformation.

**Degree-2 example:**
```
y = w1*x + w2*x² + c
```

**Key Parameters (`PolynomialFeatures`):**
| Parameter | Meaning |
|---|---|
| `degree` | Highest power of x to generate |
| `include_bias` | Whether to add a constant column of 1s |
| `interaction_only` | Only produce interaction terms (x1*x2), skip pure powers |

**Choosing degree:** Low degree → underfitting (misses curve); high degree → overfitting (fits noise). Often chosen via cross-validation.

**Sample Code:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])   # y = x^2 pattern

poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X, y)

print("Prediction for x=6:", poly_model.predict([[6]]))
```

---

## A3. Gradient Descent (Optimization Algorithm)

**Definition:** Gradient Descent is an iterative optimization algorithm used to find the values of `w` and `c` that **minimize** a cost function `J(w,c)`, by repeatedly stepping in the direction opposite to the gradient (steepest ascent).

**Update Rules (using y = wx + c form for simplicity):**

```
w := w - α * ∂J/∂w
c := c - α * ∂J/∂c
```

Where `α` (alpha) is the **learning rate** — controls step size.

**Partial derivatives for MSE cost (single feature `y = m*x + c`):**

```
∂J/∂m = -(2/m_count) * Σ x_i*(y_i - ŷ_i)
∂J/∂c = -(2/m_count) * Σ (y_i - ŷ_i)
```
(`m_count` = number of training samples, to avoid clashing with slope symbol `m`)

**General vector form:**

```
∇J(w) = (1/m) * Xᵀ(ŷ - y)
w := w - α * ∇J(w)
```

**Types of Gradient Descent:**
| Type | Data used per step | Speed/Epoch | Convergence |
|---|---|---|---|
| **Batch GD** | Entire dataset | Slow | Smooth, stable |
| **Stochastic GD (SGD)** | 1 random sample | Fast | Noisy, oscillates |
| **Mini-batch GD** | Small batch (e.g. 32/64) | Balanced | Common in deep learning |

**Key Parameters:**
| Parameter | Meaning |
|---|---|
| `learning_rate (α)` | Step size per iteration; too high → diverges, too low → very slow |
| `epochs / n_iterations` | Number of passes through the training data |
| `tolerance` | Stop early if cost improvement < tolerance |
| `momentum` | (advanced) accelerates descent in consistent directions |

**Sample Code (manual Batch Gradient Descent):**
```python
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
n = len(X)

m_slope, c_intercept = 0.0, 0.0     # y = m*x + c
alpha = 0.01
epochs = 1000

for _ in range(epochs):
    y_pred = m_slope * X + c_intercept
    dm = -(2/n) * np.sum(X * (y - y_pred))
    dc = -(2/n) * np.sum(y - y_pred)
    m_slope -= alpha * dm
    c_intercept -= alpha * dc

print(f"m={m_slope:.3f}, c={c_intercept:.3f}")
```

---

## A4. Stochastic Gradient Descent (SGD) Regressor

**Definition:** SGD updates the weights using **one training example at a time** (instead of the whole dataset), making it much faster per step and suitable for very large or streaming datasets ("online learning").

**Update Rule for a single sample `i`:**

```
w := w - α * ∇J_i(w)
```
```
∇J_i(w) = x_i * (ŷ_i - y_i)      where ŷ_i = wᵀx_i + c
```

**Key Parameters (`SGDRegressor`):**
| Parameter | Meaning |
|---|---|
| `loss` | Loss function: `squared_error`, `huber`, `epsilon_insensitive` |
| `penalty` | Regularization: `l2`, `l1`, `elasticnet`, `None` |
| `alpha` | Regularization strength constant |
| `learning_rate` | Schedule: `constant`, `optimal`, `invscaling`, `adaptive` |
| `eta0` | Initial learning rate |
| `max_iter` | Maximum number of epochs |
| `tol` | Stopping tolerance |

**Sample Code:**
```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)     # SGD is sensitive to feature scale

sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01)
sgd_model.fit(X_scaled, y)

print("Weight (w):", sgd_model.coef_)
print("Intercept (c):", sgd_model.intercept_)
```

---

## A5. Ridge Regression (L2 Regularization)

**Definition:** Ridge Regression is linear regression with an added **L2 penalty term** (sum of squared weights) that shrinks coefficients toward zero to reduce overfitting and handle multicollinearity — coefficients shrink but rarely become exactly zero.

**Hypothesis:** same as linear regression, `ŷ = wᵀx + c`

**Cost Function:**

```
J(w,c) = (1/2m) * Σ (ŷ_i - y_i)²  +  λ * Σ wj²
```

- `λ` (lambda), called `alpha` in sklearn — controls regularization strength.
- `λ = 0` → identical to ordinary linear regression.
- Larger `λ` → smaller weights → simpler, more biased, lower-variance model.

**Closed-form solution:**
```
w = (XᵀX + λI)⁻¹ Xᵀy
```

**Key Parameters (`Ridge`):**
| Parameter | Meaning |
|---|---|
| `alpha` | Regularization strength (λ) |
| `solver` | `auto`, `svd`, `cholesky`, `lsqr`, `sag`, `saga` |
| `fit_intercept` | Whether to fit `c` |
| `max_iter` | Max iterations (for iterative solvers) |

**Sample Code:**
```python
from sklearn.linear_model import Ridge
import numpy as np

X = np.array([[1, 1], [2, 2], [3, 2.5], [4, 5], [5, 4]])
y = np.array([3, 5, 6, 9, 8])

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)

print("Weights (w):", ridge_model.coef_)
print("Intercept (c):", ridge_model.intercept_)
```

---

## A6. Lasso Regression (L1 Regularization)

**Definition:** Lasso (Least Absolute Shrinkage and Selection Operator) Regression uses an **L1 penalty** (sum of absolute weights), which can shrink some coefficients to **exactly zero**, effectively performing automatic feature selection.

**Cost Function:**

```
J(w,c) = (1/2m) * Σ (ŷ_i - y_i)²  +  λ * Σ |wj|
```

**Ridge vs Lasso:**
| Aspect | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty term | `Σ wj²` | `Σ \|wj\|` |
| Coefficient shrinkage | Toward zero, rarely exactly 0 | Can be exactly 0 |
| Feature selection | No | Yes (sparse model) |
| Best when | Many correlated, all-relevant features | A few features truly matter |

**Key Parameters (`Lasso`):**
| Parameter | Meaning |
|---|---|
| `alpha` | Regularization strength (λ) |
| `max_iter` | Max number of iterations |
| `selection` | `cyclic` or `random` order of coefficient updates |
| `tol` | Optimization tolerance |

**Sample Code:**
```python
from sklearn.linear_model import Lasso
import numpy as np

X = np.array([[1, 1], [2, 2], [3, 2.5], [4, 5], [5, 4]])
y = np.array([3, 5, 6, 9, 8])

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)

print("Weights (w):", lasso_model.coef_)   # some may be exactly 0
print("Intercept (c):", lasso_model.intercept_)
```

---

## A7. Random Forest Regressor

**Definition:** An **ensemble learning** method that builds many independent decision trees on random subsets of data (bagging = Bootstrap Aggregating) and random subsets of features, then **averages** their predictions to produce a more accurate and stable result than any single tree.

**Math:**

```
ŷ = (1/N) * Σ Tree_i(x)      for i = 1 to N (number of trees)
```

Each `Tree_i` is trained on a bootstrap sample (random sampling **with replacement**) of the training data.

**Key Parameters (`RandomForestRegressor`):**
| Parameter | Meaning |
|---|---|
| `n_estimators` | Number of trees in the forest |
| `max_depth` | Max depth of each tree |
| `min_samples_split` | Min samples required to split a node |
| `min_samples_leaf` | Min samples required at a leaf |
| `max_features` | Number of features considered per split |
| `bootstrap` | Whether bootstrap sampling is used |
| `random_state` | Seed for reproducibility |

**Sample Code:**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X = np.random.rand(100, 3) * 10
y = 3*X[:,0] + 2*X[:,1] - X[:,2] + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("Feature Importances:", rf_reg.feature_importances_)
```

---

# 10. Regression Evaluation Metrics (Full Formulas)

**Notation:** `y_i` = actual value, `ŷ_i` = predicted value, `ȳ` = mean of actual values, `m` = number of samples, `k` = number of features.

### 10.1 Mean Absolute Error (MAE)
Average of absolute differences between actual and predicted values. Robust to outliers.
```
MAE = (1/m) * Σ |y_i - ŷ_i|
```

### 10.2 Mean Squared Error (MSE)
Average of squared differences. Penalizes larger errors more heavily than MAE.
```
MSE = (1/m) * Σ (y_i - ŷ_i)²
```

### 10.3 Root Mean Squared Error (RMSE)
Square root of MSE — same unit as `y`, easier to interpret.
```
RMSE = √MSE = √[ (1/m) * Σ (y_i - ŷ_i)² ]
```

### 10.4 R² Score (Coefficient of Determination)
Proportion of variance in `y` explained by the model. Ranges (−∞, 1]; 1 = perfect fit, 0 = model as good as predicting the mean.
```
R² = 1 - ( Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)² )
   = 1 - (SS_res / SS_tot)
```

### 10.5 Adjusted R²
Penalizes R² for adding features that don't improve the model — useful for comparing models with different numbers of features.
```
Adjusted R² = 1 - [ (1 - R²) * (m - 1) / (m - k - 1) ]
```

### 10.6 Mean Absolute Percentage Error (MAPE)
Error expressed as a percentage — scale independent.
```
MAPE = (100/m) * Σ | (y_i - ŷ_i) / y_i |
```

**Quick guide — when to use which:**
| Metric | Sensitive to Outliers | Interpretability | Use When |
|---|---|---|---|
| MAE | No | High (same unit as y) | Outliers should not dominate |
| MSE | Yes (squares errors) | Medium (squared unit) | Want to penalize large errors |
| RMSE | Yes | High (same unit as y) | Standard general-purpose metric |
| R² | Yes | High (0–1 scale) | Explain variance captured |
| Adjusted R² | Yes | High | Comparing models with different feature counts |
| MAPE | Yes (blows up if y≈0) | High (%) | Comparing errors across different scales |

---

# PART B — CLASSIFICATION ALGORITHMS

---

## B1. Logistic Regression

**Definition:** Despite the name, Logistic Regression is a **classification** algorithm. It estimates the **probability** that an input belongs to a particular class, using the sigmoid (logistic) function to map any real-valued number into the range (0, 1).

**Step 1 — Linear combination (same structure as linear regression):**

```
z = w1*x1 + w2*x2 + ... + wn*xn + c   =  wᵀx + c
```

**Step 2 — Sigmoid function:**

```
σ(z) = 1 / (1 + e^(-z))
```

**Step 3 — Probability & decision:**

```
P(y=1|x) = σ(wᵀx + c)
predict class 1  if P(y=1|x) ≥ 0.5   (i.e., z ≥ 0)
predict class 0  otherwise
```

**Cost Function — Binary Cross-Entropy (Log Loss):**

```
J(w,c) = -(1/m) * Σ [ y_i*log(ŷ_i) + (1-y_i)*log(1-ŷ_i) ]
```

This is minimized using Gradient Descent since there's no closed-form solution (unlike linear regression).

**Odds & Log-Odds (Logit):**
```
odds = P / (1-P)
logit(P) = log(P / (1-P)) = wᵀx + c
```

**Key Parameters (`LogisticRegression`):**
| Parameter | Meaning |
|---|---|
| `penalty` | Regularization: `l1`, `l2`, `elasticnet`, `none` |
| `C` | Inverse regularization strength (smaller C = stronger regularization) |
| `solver` | `lbfgs`, `liblinear`, `saga`, `newton-cg` |
| `max_iter` | Max iterations for convergence |
| `multi_class` | `ovr` (one-vs-rest) or `multinomial` |

**Sample Code:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(C=1.0, solver='lbfgs')
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## B2. Decision Tree Classifier

**Definition:** A Decision Tree is a flowchart-like model that splits the dataset into branches based on feature values, forming a tree where each internal node represents a "test" on a feature, each branch is an outcome, and each leaf node represents a class label.

**Splitting Criteria:**

**Gini Impurity** (used by CART, default in sklearn):
```
Gini = 1 - Σ (p_i)²        for each class i, p_i = proportion of class i in the node
```
`Gini = 0` → perfectly pure node (all same class).

**Entropy** (used in ID3/C4.5):
```
Entropy = - Σ p_i * log2(p_i)
```

**Information Gain** (used with entropy):
```
IG = Entropy(parent) - Σ ( n_child / n_parent ) * Entropy(child)
```

The algorithm greedily picks the feature + threshold that gives the **highest Information Gain** (or lowest Gini) at each split, and recurses until a stopping condition (max depth, min samples, pure node) is met.

**Key Parameters (`DecisionTreeClassifier`):**
| Parameter | Meaning |
|---|---|
| `criterion` | `gini` or `entropy` |
| `max_depth` | Maximum depth of tree (controls overfitting) |
| `min_samples_split` | Min samples needed to split a node |
| `min_samples_leaf` | Min samples required in a leaf node |
| `max_features` | Number of features considered for best split |
| `ccp_alpha` | Complexity parameter for pruning |

**Sample Code:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## B3. Random Forest Classifier

**Definition:** An ensemble of many decision trees, each trained on a random bootstrap sample with a random subset of features considered at each split. The final prediction is determined by **majority vote** across all trees, which reduces variance and overfitting compared to a single decision tree.

**Math (Majority Voting):**

```
ŷ = mode( Tree_1(x), Tree_2(x), ..., Tree_N(x) )
```

**Key Parameters (`RandomForestClassifier`):**
| Parameter | Meaning |
|---|---|
| `n_estimators` | Number of trees |
| `criterion` | `gini` or `entropy` |
| `max_depth` | Max depth per tree |
| `max_features` | Features considered per split (default `sqrt(n)`) |
| `bootstrap` | Sample with replacement (True/False) |
| `oob_score` | Estimate accuracy using out-of-bag samples |

**Sample Code:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## B4. Naive Bayes Classifier

**Definition:** A probabilistic classifier based on **Bayes' Theorem**, which assumes all features are **conditionally independent** given the class label (a "naive" simplifying assumption). Despite this assumption rarely being true in practice, it performs remarkably well, especially for text classification.

**Bayes' Theorem:**

```
P(y|x) = [ P(x|y) * P(y) ] / P(x)
```

| Term | Meaning |
|---|---|
| `P(y\|x)` | Posterior — probability of class y given features x |
| `P(x\|y)` | Likelihood — probability of observing x given class y |
| `P(y)` | Prior — probability of class y before seeing data |
| `P(x)` | Evidence — overall probability of observing x |

**Applying the "naive" independence assumption** (for features `x1...xn`):

```
P(x|y) = P(x1|y) * P(x2|y) * ... * P(xn|y) = Π P(xi|y)
```

**Final classification rule** (denominator `P(x)` is constant across classes, so it's dropped):

```
ŷ = argmax_y  [ P(y) * Π P(xi|y) ]     for i = 1 to n
```

**Gaussian likelihood (for continuous features, GaussianNB):**

```
P(xi|y) = 1/√(2π*σy²) * exp( -(xi - μy)² / (2σy²) )
```
where `μy` and `σy²` are the mean and variance of feature `xi` within class `y`.

**Common Variants:**
| Variant | Use Case | Feature Type |
|---|---|---|
| **GaussianNB** | Continuous features | Assumes Normal distribution |
| **MultinomialNB** | Word counts / text data | Discrete counts |
| **BernoulliNB** | Binary features | 0/1 presence-absence |

**Key Parameters (`GaussianNB`):**
| Parameter | Meaning |
|---|---|
| `var_smoothing` | Small value added to variances for numerical stability |
| `priors` | Manually specify class prior probabilities |

**Sample Code:**
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## B5. Support Vector Machine (SVM)

**Definition:** SVM is a classifier that finds the **optimal separating hyperplane** that maximizes the margin (distance) between the closest points of each class ("support vectors"). It can classify non-linear data by mapping inputs into higher dimensions via the **kernel trick**.

**Hyperplane equation** (same `wᵀx + c` form seen throughout):

```
w·x + c = 0
```

**Classification decision rule:**
```
ŷ = sign(w·x + c)
```

**Margin:** the perpendicular distance between the hyperplane and the nearest data point of each class:
```
margin = 2 / ||w||
```
SVM tries to **maximize** this margin (equivalently, minimize `||w||`).

**Hard-margin optimization (linearly separable data):**
```
minimize    (1/2) ||w||²
subject to   y_i (w·x_i + c) ≥ 1    for all i
```

**Soft-margin optimization (allows some misclassification, controlled by `C`):**
```
minimize    (1/2)||w||² + C * Σ ξ_i
subject to   y_i(w·x_i + c) ≥ 1 - ξ_i,   ξ_i ≥ 0
```
`ξ_i` (xi) = slack variable allowing a point to violate the margin; `C` controls the tradeoff between maximizing margin and minimizing misclassification.

**Kernel Trick** (maps data to higher dimension implicitly, without computing the transform explicitly):
| Kernel | Formula |
|---|---|
| Linear | `K(x,x') = x·x'` |
| Polynomial | `K(x,x') = (γ*x·x' + r)^d` |
| RBF (Gaussian) | `K(x,x') = exp(-γ * \|\|x-x'\|\|²)` |
| Sigmoid | `K(x,x') = tanh(γ*x·x' + r)` |

**Key Parameters (`SVC`):**
| Parameter | Meaning |
|---|---|
| `C` | Regularization; small C = wider margin, more misclassification allowed |
| `kernel` | `linear`, `poly`, `rbf`, `sigmoid` |
| `gamma` | Kernel coefficient — controls how far a single point's influence reaches |
| `degree` | Degree of the polynomial kernel |

**Sample Code:**
```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_s, y_train)
y_pred = svm_model.predict(X_test_s)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

# 11. Classification Evaluation Metrics (Full Formulas)

### 11.1 Confusion Matrix

The foundation of all classification metrics (binary case shown):

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

- **TP:** correctly predicted positive
- **TN:** correctly predicted negative
- **FP (Type I error):** predicted positive, actually negative
- **FN (Type II error):** predicted negative, actually positive

### 11.2 Accuracy
Overall proportion of correct predictions. Misleading on imbalanced datasets.
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### 11.3 Precision
Of all predicted positives, how many are actually positive? ("How trustworthy are positive predictions?")
```
Precision = TP / (TP + FP)
```

### 11.4 Recall (Sensitivity / True Positive Rate)
Of all actual positives, how many did the model correctly identify?
```
Recall = TP / (TP + FN)
```

### 11.5 Specificity (True Negative Rate)
Of all actual negatives, how many were correctly identified?
```
Specificity = TN / (TN + FP)
```

### 11.6 F1-Score
Harmonic mean of Precision and Recall — useful when you need a balance between the two, especially with imbalanced classes.
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### 11.7 F-beta Score (generalized F1)
Weighs recall `β` times as important as precision.
```
Fβ = (1 + β²) * (Precision * Recall) / (β²*Precision + Recall)
```

### 11.8 Log Loss (Cross-Entropy Loss)
Penalizes confident wrong predictions heavily; lower is better.
```
LogLoss = -(1/m) * Σ [ y_i*log(ŷ_i) + (1-y_i)*log(1-ŷ_i) ]
```

### 11.9 ROC Curve & AUC
- **ROC curve:** plots True Positive Rate (Recall) vs. False Positive Rate at various classification thresholds.
```
FPR = FP / (FP + TN)
TPR = TP / (TP + FN)   (same as Recall)
```
- **AUC (Area Under the Curve):** ranges 0.5 (random guessing) to 1.0 (perfect classifier); measures the model's ability to distinguish between classes across all thresholds.

### 11.10 Matthews Correlation Coefficient (MCC)
A balanced measure even for imbalanced classes; ranges from -1 (total disagreement) to +1 (perfect prediction).
```
MCC = (TP*TN - FP*FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

**Quick Guide — when to use which:**
| Metric | Best Used When |
|---|---|
| Accuracy | Classes are balanced |
| Precision | False positives are costly (e.g., spam filter) |
| Recall | False negatives are costly (e.g., disease detection) |
| F1-Score | Need balance between precision & recall, imbalanced classes |
| ROC-AUC | Comparing models across all thresholds |
| Log Loss | Care about probability calibration, not just class labels |
| MCC | Highly imbalanced datasets |

**Sample Code — Computing Multiple Metrics:**
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, log_loss)

# assume y_test, y_pred, y_proba (predicted probabilities) already computed
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# For binary classification with probability outputs:
# print("ROC-AUC:", roc_auc_score(y_test, y_proba[:,1]))
# print("Log Loss:", log_loss(y_test, y_proba))
```

---

# 12. Master Comparison Tables

### Regression Algorithms
| Algorithm | Formula Basis | Handles Non-linearity | Regularized | Interpretable |
|---|---|---|---|---|
| Linear Regression | `y = wx + c` | No | No | High |
| Polynomial Regression | `y = w1x + w2x² + ... + c` | Yes | No | Medium |
| Gradient Descent | Optimizer (not a model itself) | — | — | — |
| SGD Regressor | `y = wx + c` (online updates) | No | Yes (l1/l2/elasticnet) | Medium |
| Ridge Regression | `y = wx + c` + L2 penalty | No | Yes (L2) | High |
| Lasso Regression | `y = wx + c` + L1 penalty | No | Yes (L1) | High |
| Random Forest Regressor | Average of trees | Yes | Implicit | Low |

### Classification Algorithms
| Algorithm | Formula Basis | Interpretable | Handles Non-linearity | Scale-Sensitive |
|---|---|---|---|---|
| Logistic Regression | `σ(wx+c)` | High | No (unless polynomial features) | Yes |
| Decision Tree | Gini/Entropy splits | High | Yes | No |
| Random Forest | Majority vote of trees | Low-Medium | Yes | No |
| Naive Bayes | Bayes' Theorem | Medium | No (assumes independence) | No |
| SVM | `wx + c = 0` hyperplane | Low-Medium | Yes (with kernels) | Yes |

---

# 13. Bias-Variance Tradeoff & Overfitting

- **High Bias (Underfitting):** model is too simple → poor performance on both train and test data.
  *Fix:* add more/better features, reduce regularization, use a more complex model.
- **High Variance (Overfitting):** model is too complex → excellent on train data, poor on test data.
  *Fix:* add regularization (Ridge/Lasso), prune decision trees, gather more training data, use ensembles (Random Forest), cross-validation.
- **Regularization strength** (`λ`/`alpha` in Ridge & Lasso, inverse in `C` for Logistic Regression/SVM): larger `λ` (or smaller `C`) → simpler model → more bias, less variance.

```
Total Error ≈ Bias² + Variance + Irreducible Error
```

---

*End of Cheatsheet.*
