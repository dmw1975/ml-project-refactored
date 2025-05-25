import pandas as pd

# Load your model comparison test results
tests_df = pd.read_csv('/mnt/d/ml_project_refactored/outputs/metrics/model_comparison_tests.csv')  # <-- adjust path if needed

models = sorted(list(set(tests_df['model_a']).union(set(tests_df['model_b']))))
print("All models found in statistical tests:")
for m in models:
    print(m)
