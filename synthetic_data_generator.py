'''
Synthetic Tabular Dataset Generator for Feature Selection Experiments
Created by Nikos Christakis

Affiliation:
    Institute for Advanced Modelling and Simulation, University of Nicosia, Cyprus

Contact:
    christakis.n@unic.ac.cy

Overview:
    This script generates synthetic tabular datasets with a controlled mixture of:
      - informative continuous predictors,
      - categorical predictors (derived via quantiles),
      - time-dependent predictors (trend + seasonality),
      - redundant/noise predictors,
    and supports either:
      (i) binary classification targets (Y in {0,1}), or
      (ii) continuous regression targets.

Core functions:
    - create_synthetic_dataset(...): generates (X, Y) as a pandas DataFrame/Series.
    - plot_and_export_correlations(...): exports correlation statistics and saves plots.

Correlation control (important practical note):
    The parameter `correlation_strength` controls the intended dependence between each
    informative feature and the target. In finite samples and after iterative adjustments
    (especially in the classification case), the *achieved* empirical correlation may be
    slightly lower than the value requested. In practice, if you require an achieved
    correlation of approximately r (e.g., r ≈ 0.30), you may need to set
    `correlation_strength` to a slightly higher value (for example, 0.4) and then
    verify the realised correlation in the generated dataset (e.g., via
    plot_and_export_correlations).

Reproducibility:
    All randomness is controlled via `random_state` (NumPy seed is set internally).

Outputs:
    When executed as a script, the default example writes a CSV:
        synthetic_feature_selection_dataset.csv
    containing features and the target column Y.
'''
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def create_synthetic_dataset(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_categorical=2,
    n_time_dependent=1,
    n_low_variation=[0, 3],
    variation_std=1e-2,
    low_var=False, #for low variation of a feature
    correlation_strength=0.2,
    noise=0.1,
    random_state=42,
    include_feature_types=True,
    classification = True
):
	#stop if feature numbers don't match
    if n_features < (n_informative + n_categorical + n_time_dependent):
        exit("Error: Total features must be at least the sum of informative, categorical, and time-dependent features.")

    n_feat = n_informative + n_categorical + n_time_dependent #to incorprate all features in X array		
    np.random.seed(random_state)
    if classification:
        X_base, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative,
        n_informative=n_informative,
        n_redundant=0, # must add, default is 2
        n_classes=2,  # or any number of categories
        random_state=random_state
        )
    else:
        X_base, y = make_regression(
        n_samples=n_samples,
        n_features=n_informative,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
        )
    # Correlate X's with Y 
    # use standrardized y
    y_std = (y - y.mean()) / y.std()   
    # Define desired correlations (between 0 and 1)
    desired_corr = correlation_strength  # use the passed parameter
    
    #Separate the correlation of informative features
    #for the case of classification
    if classification:
		# Get class proportions
	    y_binary = y
	    max_iterations = 500
	    tolerance = 1.e-10
	    p1 = np.mean(y_binary)  # proportion of class 1
	    p0 = 1 - p1  # proportion of class 0
    
		# Check if desired correlation is achievable
	    max_possible = np.sqrt(p0 * p1) / (0.5 * (p0 + p1))
	    if abs(desired_corr) > max_possible * 0.95:  # 95% of theoretical max
		    print(f"Warning: Desired correlation {desired_corr:.3f} may be too high.")
		    print(f"Theoretical maximum: ±{max_possible:.3f}")
		    desired_correlation = np.sign(desired_corr) * max_possible * 0.9
    
	    for i in range(X_base.shape[1]):
		    x = np.random.normal(0, 1, size=n_samples)
        
		    # Use smaller adjustment steps for better convergence
		    learning_rate = 0.5  # Damping factor
        
		    for iteration in range(max_iterations):
			    current_corr = np.corrcoef(y_binary, x)[0, 1]
            
			    if abs(current_corr - desired_corr) < tolerance:
				    break
                
			    # Calculate required adjustment with damping
			    error = desired_corr - current_corr
            
			    # Calculate means
			    mean_0 = np.mean(x[y_binary == 0])
			    mean_1 = np.mean(x[y_binary == 1])
			    overall_std = np.std(x)
            
			    # Required mean difference for target correlation
			    target_mean_diff = desired_corr * overall_std / np.sqrt(p0 * p1)
			    current_mean_diff = mean_1 - mean_0
            
				# Apply damped adjustment
			    adjustment = learning_rate * (target_mean_diff - current_mean_diff)
            
				# Adjust means while preserving overall mean
			    overall_mean = np.mean(x)
			    x[y_binary == 0] -= p1 * adjustment
			    x[y_binary == 1] += p0 * adjustment
            
				# Re-standardize to maintain unit variance
			    x = (x - np.mean(x)) / np.std(x)
        
		    X_base[:, i] = x
    else:
    
    
        # Reconstruct X_base and
        # adjust each informative feature's linear correlation with y
        for i in range(X_base.shape[1]):
           noise_component = np.random.normal(0, 1, size=n_samples)
           X_base[:, i] = desired_corr * y_std + np.sqrt(1 - desired_corr**2) * noise_component

    feature_names = [f'X{i+1}' for i in range(n_features)]
    X = pd.DataFrame(X_base, columns=feature_names[:n_informative])
    
    # Add categorical features with weak correlation to Y
    for i in range(n_informative, n_informative+n_categorical):
        cat = pd.qcut(y + np.random.normal(0, 10, size=n_samples), q=2, labels=False)
        X[feature_names[i]] = cat
       
    
    # Add time-dependent features
    time = np.arange(n_samples)
    for i in range(n_informative+n_categorical, n_feat):
        trend = time * np.random.uniform(0.001, 0.01)
        seasonal = 10 * np.sin(time / 20)
        X[feature_names[i]] = trend + seasonal + np.random.normal(0, 0.5, n_samples)


    # Add redundant + noise features
    for i in range(n_feat, n_features):
        X[feature_names[i]] = np.random.normal(0, 1, size=n_samples)

    # Overwrite certain columns with low variation 
    #if low variation is on
    if low_var:
        for idx in n_low_variation:
            if idx < len(X.columns):
                X.iloc[:, idx] = np.random.normal(0, variation_std, size=n_samples)

    
    # Add feature type transformations
    if include_feature_types and n_informative >= 3:
        X['X_bin'] = (X.iloc[:, 0] > X.iloc[:, 0].mean()).astype(int)
        X['X_poly'] = X.iloc[:, 1] ** 2
        X['X_log'] = np.log(np.abs(X.iloc[:, 2]) + 1)
        X['X_exp'] = np.exp(X.iloc[:, 2] / 10)

    return X, pd.Series(y, name='Y')

def plot_and_export_correlations(X, y, export_path="correlation_stats.csv"):
    full_data = pd.concat([X, y], axis=1)
    corr = full_data.corr()
    corr_with_y = corr['Y'].drop('Y').sort_values(ascending=False)
    p_values = X.apply(lambda col: pearsonr(col, y)[1])

    stats = pd.DataFrame({
        'Correlation': corr_with_y,
        'P-Value': p_values[corr_with_y.index]
    }).sort_values(by='Correlation', ascending=False)

    stats.to_csv(export_path)
    print(f"Correlation stats exported to {export_path}")

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('Correlation_matrix.png', dpi=600)
    plt.close()

    # Plot correlation bar graph
    plt.figure(figsize=(10, 6))
    stats['Correlation'].plot(kind='bar')
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel('Correlation with Y')
    plt.title('Feature Correlation with Y')
    plt.tight_layout()
    plt.savefig('Correlation_graph.png', dpi=600)
    plt.close()

if __name__ == "__main__":
    X, y = create_synthetic_dataset(
        n_samples=10000,
        n_features=20,
        n_informative=2,
        n_categorical=0,
        n_time_dependent=0,
        n_low_variation=[0, 3],
        variation_std=1e-1,
        low_var=False,
        correlation_strength=0.4,
        include_feature_types=False
    )
    X.join(y).to_csv("synthetic_feature_selection_dataset.csv", index=False)
    #plot_and_export_correlations(X, y)
