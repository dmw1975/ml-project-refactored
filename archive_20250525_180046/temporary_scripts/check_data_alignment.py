import pandas as pd
from pathlib import Path
from config import settings

# Load tree models CSV
tree_csv = pd.read_csv(Path(settings.DATA_DIR) / 'raw' / 'combined_df_for_tree_models.csv')
print(f'Tree models CSV shape: {tree_csv.shape}')

# Load linear models CSV  
linear_csv = pd.read_csv(Path(settings.DATA_DIR) / 'raw' / 'combined_df_for_ml_models.csv')
print(f'Linear models CSV shape: {linear_csv.shape}')

# Load scores
scores = pd.read_csv(Path(settings.DATA_DIR) / 'raw' / 'score.csv')
print(f'Scores CSV shape: {scores.shape}')

# Check issuer_name columns
print(f'\nTree CSV has issuer_name: {"issuer_name" in tree_csv.columns}')
print(f'Linear CSV has issuer_name: {"issuer_name" in linear_csv.columns}')
print(f'Scores CSV has issuer_name: {"issuer_name" in scores.columns}')

# Check for duplicates or missing values
if 'issuer_name' in tree_csv.columns:
    print(f'\nTree CSV unique issuer_names: {tree_csv["issuer_name"].nunique()}')
    print(f'Tree CSV duplicated issuer_names: {tree_csv["issuer_name"].duplicated().sum()}')
    print(f'Tree CSV missing issuer_names: {tree_csv["issuer_name"].isna().sum()}')

if 'issuer_name' in scores.columns:
    print(f'\nScores unique issuer_names: {scores["issuer_name"].nunique()}')
    print(f'Scores duplicated issuer_names: {scores["issuer_name"].duplicated().sum()}')
    print(f'Scores missing issuer_names: {scores["issuer_name"].isna().sum()}')

# Now check what happens with the intersection
if 'issuer_name' in tree_csv.columns and 'issuer_name' in scores.columns:
    tree_issuers = set(tree_csv['issuer_name'])
    score_issuers = set(scores['issuer_name'])
    
    common_issuers = tree_issuers.intersection(score_issuers)
    print(f'\nCommon issuers between tree CSV and scores: {len(common_issuers)}')
    print(f'Issuers only in tree CSV: {len(tree_issuers - score_issuers)}')
    print(f'Issuers only in scores: {len(score_issuers - tree_issuers)}')
    
    # Show some examples of mismatches
    if len(tree_issuers - score_issuers) > 0:
        print('\nExamples of issuers only in tree CSV:')
        for issuer in list(tree_issuers - score_issuers)[:5]:
            print(f'  - {issuer}')
    
    if len(score_issuers - tree_issuers) > 0:
        print('\nExamples of issuers only in scores:')
        for issuer in list(score_issuers - tree_issuers)[:5]:
            print(f'  - {issuer}')

# Check how linear models handle the join
print('\n--- Linear Model Data Loading ---')
# Simulate what linear models do
features = linear_csv
scores_df = scores
print(f'Features shape: {features.shape}')
print(f'Scores shape: {scores_df.shape}')

# Linear models seem to join differently - they don't use issuer_name index
# They likely rely on row order matching
print(f'After direct join (assuming row order): {features.shape[0]} samples')

# Check if tree models are losing samples during join
print('\n--- Tree Model Data Loading (Simulated) ---')
tree_df = tree_csv.copy()
scores_copy = scores.copy()

if 'issuer_name' in tree_df.columns:
    tree_df = tree_df.set_index('issuer_name')
    
if 'issuer_name' in scores_copy.columns:
    scores_copy = scores_copy.set_index('issuer_name')
    
common_indices = tree_df.index.intersection(scores_copy.index)
print(f'After intersection join: {len(common_indices)} samples')
print(f'Lost samples: {len(tree_df) - len(common_indices)}')