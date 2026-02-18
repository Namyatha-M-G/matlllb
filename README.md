"""
AI/ML and Statistical Analysis Script
=====================================================
Executes the 5 required mathematical and statistical operations:
1. Correlation and Regression (Simple and Multiple Linear Regression)
2. Vector Space Operations (Linear Independence Check)
3. Eigenvalues and Eigenvectors
4. System of Linear Equations (Solution Classification)
5. Probability Distributions (Visualization and Joint Probability)

Uses NumPy, Pandas, SciPy, and Statsmodels.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy import linalg
from scipy import stats
import matplotlib.pyplot as plt
import warnings

# Suppress minor warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- DATA EXTRACTION AND SETUP (n=32 data points) ---
# Data corresponds to the Motor Trend Car Road Tests dataset (mtcars)
# MPG: Miles/(US) gallon, HP: Horsepower, WT: Weight (1000 lbs), AM: Transmission (0=Auto, 1=Manual)
data = {
    'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4],
    'cyl': [6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4],
    'disp': [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1, 120.3, 203.7, 301.0, 350.0, 400.0, 79.0, 120.3, 95.1, 351.0, 145.0, 301.0, 121.0],
    'hp': [110, 110, 93, 110, 175, 105, 245, 66, 110, 100, 110, 180, 180, 180, 205, 215, 230, 52, 65, 97, 109, 150, 150, 180, 200, 66, 91, 113, 264, 175, 335, 109],
    'drat': [3.90, 3.90, 3.85, 3.08, 3.15, 2.76, 3.21, 3.69, 3.92, 3.92, 3.92, 3.07, 3.07, 3.07, 2.93, 3.00, 3.23, 4.08, 4.93, 3.70, 4.22, 3.78, 3.15, 3.73, 4.22, 4.11, 4.43, 3.77, 4.22, 3.62, 3.54, 4.11],
    'wt': [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440, 3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835, 2.465, 3.520, 3.435, 3.840, 4.070, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780],
    'qsec': [16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20.00, 22.90, 18.30, 18.90, 17.40, 17.60, 18.00, 17.98, 17.82, 17.42, 19.47, 18.52, 19.90, 20.01, 16.89, 17.30, 15.41, 17.05, 18.90, 16.70, 16.90, 14.50, 15.50, 14.60, 18.60],
    'vs': [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
    'am': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    'gear': [4, 4, 4, 3, 3, 3, 3, 3, 4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 5, 5, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 4],
    'carb': [4, 4, 1, 1, 2, 1, 4, 2, 2, 4, 4, 3, 3, 3, 4, 4, 4, 1, 2, 2, 1, 2, 2, 4, 2, 2, 2, 2, 4, 6, 8, 2]
}
df = pd.DataFrame(data)

print("="*80)
print("AI/ML PROJECT ANALYSIS RESULTS")
print("="*80)

# ----------------------------------------------------------------------
# 1. CORRELATION, LINEAR REGRESSION (Simple and Multiple)
# ----------------------------------------------------------------------
print("1.) CORRELATION AND REGRESSION\n")

# 1a. Correlation Analysis
print("--- Pearson Correlation (MPG, HP, WT, AM) ---")
corr_matrix = df[['mpg', 'hp', 'wt', 'am']].corr()
print(corr_matrix.round(3))
print(f"\nObservation: MPG is highly negatively correlated with Horsepower (-0.776) and Weight (-0.868).")

# 1b. Simple Linear Regression (MPG vs Weight)
print("\n--- Simple Linear Regression: mpg ~ wt ---")
X_simple = sm.add_constant(df['wt'])
y = df['mpg']
model_simple = sm.OLS(y, X_simple).fit()
print(model_simple.summary().tables[1]) # Displaying coefficient table
print(f"Model: MPG = {model_simple.params['const']:.2f} + {model_simple.params['wt']:.2f} * Weight")


# 1c. Multiple Linear Regression (MPG vs HP, Weight, and Transmission)
print("\n--- Multiple Linear Regression: mpg ~ hp + wt + am ---")
X_multi = df[['hp', 'wt', 'am']]
X_multi = sm.add_constant(X_multi)
model_multi = sm.OLS(y, X_multi).fit()
print(model_multi.summary().tables[1])
print(f"Model: MPG = {model_multi.params['const']:.2f} + {model_multi.params['hp']:.3f}*HP +
       {model_multi.params['wt']:.2f}*Weight + {model_multi.params['am']:.2f}*Transmission(Manual=1)")
print("-" * 80)

# ----------------------------------------------------------------------
# 2. VECTOR SPACE OPERATIONS (Linear Dependence/Independence)
# ----------------------------------------------------------------------
print("2.) VECTOR SPACE OPERATIONS\n")

# Vectors of Horsepower, Displacement, and Weight
V_hp = df['hp'].values
V_disp = df['disp'].values
V_wt = df['wt'].values

# Check Linear Independence by calculating the rank of the matrix A
A = np.array([V_hp, V_disp, V_wt]).T # 32x3 Matrix
rank_A = np.linalg.matrix_rank(A)
n_vectors = A.shape[1]

print("--- Linear Dependence/Independence Check ---")
print(f"Number of vectors: {n_vectors}")
print(f"Rank of the matrix formed by [HP, DISP, WT]: {rank_A}")

if rank_A == n_vectors:
    print("Classification: Linearly Independent (Rank = Number of Vectors).")
else:
    print("Classification: Linearly Dependent (Rank < Number of Vectors).")

print("-" * 80)

# ----------------------------------------------------------------------
# 3. EVALUATION EIGENVALUES AND EIGENVECTORS
# ----------------------------------------------------------------------
print("3.) EIGENVALUES AND EIGENVECTORS\n")

# Compute the Covariance matrix (C) for the most correlated subset
V = df[['disp', 'hp', 'wt']]
C = np.cov(V, rowvar=False)

print("--- Covariance Matrix C (3x3) ---")
print(pd.DataFrame(C.round(2), index=['disp', 'hp', 'wt'], columns=['disp', 'hp', 'wt']))

# Calculate Eigenvalues (L) and Eigenvectors (V)
L, V = linalg.eig(C) 

print("\n--- Eigenvalues (L) ---")
print(L.round(2))
print("\n--- Eigenvectors (V) (Each column V[:, i] corresponds to L[i]) ---")
print(V.round(3))

print(f"\nInterpretation: The largest eigenvalue is {L[0]:.2f}, indicating the direction of maximum variance in the data.")
print("-" * 80)
0

# ----------------------------------------------------------------------
# 4. DETERMINE NUMBER OF SOLUTIONS (Classification)
# ----------------------------------------------------------------------
print("4.) DETERMINING THE NUMBER OF SOLUTIONS AND CLASSIFICATION\n")

# Rouché-Capelli Theorem: Rank(A) vs Rank([A|B]) vs Number of Variables (n)

# --- Case 1: Unique Solution (Rank(A) = Rank([A|B]) = n)
A1 = np.array([[2, 1, -1], [1, 3, 1], [3, 0, 2]])
B1 = np.array([1, 5, 4])
rank_A1 = np.linalg.matrix_rank(A1)
rank_AB1 = np.linalg.matrix_rank(np.column_stack((A1, B1)))
n_vars = A1.shape[1]

print("--- Case 1: Unique Solution ---")
print(f"Rank(A): {rank_A1}, Rank([A|B]): {rank_AB1}, Variables (n): {n_vars}")
if rank_A1 == rank_AB1 == n_vars:
    X1 = linalg.solve(A1, B1)
    print(f"Classification: Unique Solution.")
    print(f"Solution X: {X1.round(2)}")


# --- Case 2: No Solution (Rank(A) < Rank([A|B]))
A2 = np.array([[1, 1, 1], [1, 1, 1], [1, 2, 2]])
B2 = np.array([1, 5, 4]) # Inconsistent B vector
rank_A2 = np.linalg.matrix_rank(A2)
rank_AB2 = np.linalg.matrix_rank(np.column_stack((A2, B2)))

print("\n--- Case 2: No Solution ---")
print(f"Rank(A): {rank_A2}, Rank([A|B]): {rank_AB2}, Variables (n): {n_vars}")
if rank_A2 < rank_AB2:
    X2_approx = linalg.lstsq(A2, B2, rcond=None)[0]
    print("Classification: No Solution (Inconsistent System).")
    print(f"Best Approximate Solution (Least Squares): {X2_approx.round(2)}")
 

# --- Case 3: Infinite Solutions (Rank(A) = Rank([A|B]) < n)
A3 = np.array([[1, 2, -1], [2, 4, -2], [3, 6, -3]])
B3 = np.array([1, 2, 3]) # Consistent but dependent rows
rank_A3 = np.linalg.matrix_rank(A3)
rank_AB3 = np.linalg.matrix_rank(np.column_stack((A3, B3)))

print("\n--- Case 3: Infinite Solutions ---")
print(f"Rank(A): {rank_A3}, Rank([A|B]): {rank_AB3}, Variables (n): {n_vars}")

if rank_A3 == rank_AB3 and rank_A3 < n_vars:
    free_vars = n_vars - rank_A3
    print(f"Classification: Infinite Solutions (Dependent System).")
    print(f"Number of free variables: {free_vars}.")
print("-" * 80)


# ----------------------------------------------------------------------
# 5. PROBABILITY DISTRIBUTIONS (Visualization and Joint Probability)
# ----------------------------------------------------------------------
print("5.) PROBABILITY DISTRIBUTIONS\n")

# 5a. Joint Probability Distribution (Discrete variables from data)
# Transmission (am: 0/1) vs Cylinders (cyl: 4/6/8)
joint_probability = pd.crosstab(df['am'], df['cyl'], normalize='all')

print("--- Joint Probability P(Transmission AND Cylinders) ---")
print("This shows P(A ∩ B) for each combination, normalized by the total 32 observations:")
print(joint_probability.round(4))
# 5b. Visualize Distributions (Binomial, Poisson, Uniform, Exponential)
plt.style.use('grayscale')
plt.figure(figsize=(15, 10))
plt.suptitle("Probability Distributions Visualization", fontsize=16)

# Binomial
x_binom = np.arange(0, 11)
pmf_binom = stats.binom.pmf(x_binom, 10, 0.5)
plt.subplot(2, 2, 1)
plt.bar(x_binom, pmf_binom, edgecolor='black')
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.xlabel('k (Successes)')

# Poisson
x_poisson = np.arange(0, 10)
pmf_poisson = stats.poisson.pmf(x_poisson, 3)
plt.subplot(2, 2, 2)
plt.bar(x_poisson, pmf_poisson, edgecolor='black')
plt.title('Poisson Distribution ($\mu=3$)')
plt.xlabel('k (Events)')

# Uniform
x_unif = np.linspace(0, 10, 100)
pdf_unif = stats.uniform.pdf(x_unif, loc=2, scale=6) # Range [2, 8]
plt.subplot(2, 2, 3)
plt.plot(x_unif, pdf_unif, 'k-', lw=2)
plt.fill_between(x_unif, 0, pdf_unif, alpha=0.6)
plt.title('Uniform Distribution (loc=2, scale=6)')
plt.xlabel('Value (x)')

# Exponential
x_exp = np.linspace(0, 10, 100)
pdf_exp = stats.expon.pdf(x_exp, scale=2) # Rate lambda=0.5 (1/mean)
plt.subplot(2, 2, 4)
plt.plot(x_exp, pdf_exp, 'k-', lw=2)
plt.fill_between(x_exp, 0, pdf_exp, alpha=0.6)
plt.title('Exponential Distribution ($\lambda=0.5$)')
plt.xlabel('Time (t)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n" + "="*80)
print("Analysis Complete. Check your Spyder Plots window for visualizations.")
print("="*80)
