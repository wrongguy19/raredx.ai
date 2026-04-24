# ============================================
# RARE DISEASE AI PROJECT
# Step 3b: HPO Symptom Integration
# ============================================
# Run AFTER step3_model_training.py
# Run BEFORE step4_doctor_interface.py
# ============================================

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing    import normalize
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 55)
print("  STEP 3b: HPO SYMPTOM INTEGRATION")
print("=" * 55)

# ============================================
# LOAD HPO ANNOTATION FILE
# ============================================
print("\n[1/5] Loading HPO annotations...")

# Read the HPO annotation file
# Skip comment lines starting with #
hpo_raw = pd.read_csv(
    'phenotype.hpoa',
    sep         = '\t',
    comment     = '#',
    low_memory  = False,
    header      = 0
)

print(f"   OK Raw HPO rows     : {len(hpo_raw):,}")
print(f"   OK Columns          : {list(hpo_raw.columns[:6])}")

# ============================================
# FILTER TO ORPHANET DISEASES ONLY
# ============================================
print("\n[2/5] Filtering to Orphanet diseases...")

# HPO file has disease IDs like ORPHA:58, OMIM:203450
# We need ORPHA entries to match our OrphaCode
hpo_orpha = hpo_raw[
    hpo_raw.iloc[:, 0].astype(str).str.startswith('ORPHA')
].copy()

# Extract numeric OrphaCode
hpo_orpha['OrphaCode'] = hpo_orpha.iloc[:, 0]\
    .str.replace('ORPHA:', '', regex=False)\
    .astype(int)

# Get HPO term column (usually column named 'hpo_id' or similar)
# Detect it automatically
hpo_col = None
for col in hpo_orpha.columns:
    if hpo_orpha[col].astype(str).str.startswith('HP:').sum() > 100:
        hpo_col = col
        break

if hpo_col is None:
    # Try by position — HPO ID is usually 3rd or 4th column
    for i in range(2, 6):
        if hpo_raw.iloc[:, i].astype(str).str.startswith('HP:').sum() > 100:
            hpo_col = hpo_raw.columns[i]
            break

print(f"   OK HPO term column  : '{hpo_col}'")
print(f"   OK Orphanet entries : {len(hpo_orpha):,}")
print(f"   OK Unique diseases  : {hpo_orpha['OrphaCode'].nunique():,}")
print(f"   OK Unique HPO terms : {hpo_orpha[hpo_col].nunique():,}")

# ============================================
# BUILD HPO FEATURE MATRIX
# ============================================
print("\n[3/5] Building HPO feature matrix...")
print("      (Top 200 most common symptom terms)")

# Keep top 200 most frequent HPO terms
# (avoids a massive sparse matrix)
top200_hpo = hpo_orpha[hpo_col]\
    .value_counts().head(200).index.tolist()

hpo_filtered = hpo_orpha[
    hpo_orpha[hpo_col].isin(top200_hpo)
][['OrphaCode', hpo_col]].copy()

# Pivot: rows = disease, cols = HPO terms
hpo_matrix = hpo_filtered\
    .drop_duplicates()\
    .pivot_table(
        index      = 'OrphaCode',
        columns    = hpo_col,
        values     = hpo_col,
        aggfunc    = 'count',
        fill_value = 0
    ).clip(0, 1)

# Rename columns to hpo_ prefix
hpo_matrix.columns = [
    'hpo_' + c.replace(':', '_')
    for c in hpo_matrix.columns
]

print(f"   OK HPO matrix shape : "
      f"{hpo_matrix.shape[0]:,} diseases × {hpo_matrix.shape[1]:,} symptoms")

# ============================================
# MERGE WITH EXISTING MASTER FEATURES
# ============================================
print("\n[4/5] Merging HPO into master feature table...")

df_master = pd.read_csv('output_master_features.csv')
with open('output_feature_cols.pkl', 'rb') as f:
    feature_cols_old = pickle.load(f)

# Merge HPO matrix
df_enhanced = df_master.merge(
    hpo_matrix.reset_index(),
    on    = 'OrphaCode',
    how   = 'left'
)

# Fill missing HPO values with 0
hpo_cols = [c for c in df_enhanced.columns
            if c.startswith('hpo_')]
df_enhanced[hpo_cols] = df_enhanced[hpo_cols].fillna(0)

# New feature columns = old + HPO
feature_cols_new = feature_cols_old + hpo_cols

print(f"   OK Old features     : {len(feature_cols_old):,}")
print(f"   OK HPO features     : {len(hpo_cols):,}")
print(f"   OK Total features   : {len(feature_cols_new):,}")
print(f"   OK Diseases covered : "
      f"{(df_enhanced[hpo_cols].sum(axis=1) > 0).sum():,} "
      f"have at least one HPO term")

# ============================================
# REBUILD SIMILARITY INDEX WITH HPO
# ============================================
print("\n[5/5] Rebuilding similarity index with HPO features...")

X_raw = np.nan_to_num(
    df_enhanced[feature_cols_new].values.astype(np.float32),
    nan=0.0, posinf=0.0, neginf=0.0
)
X_norm = normalize(X_raw, norm='l2')

print(f"   OK New feature matrix : {X_norm.shape}")
print(f"   OK NaN count          : {np.isnan(X_norm).sum()}")

# Quick evaluation — tier match with HPO
n          = len(df_enhanced)
rng        = np.random.RandomState(42)
indices    = rng.permutation(n)
train_idx  = indices[:int(0.8 * n)]
test_idx   = indices[int(0.8 * n):]

sim_matrix  = cosine_similarity(X_norm[test_idx], X_norm[train_idx])
top5_match  = 0
top1_match  = 0

for i in range(len(test_idx)):
    sims       = sim_matrix[i]
    top5_j     = sims.argsort()[-5:][::-1]
    top1_j     = sims.argsort()[-1:][::-1]
    test_tier  = df_enhanced.iloc[test_idx[i]]['urgency_tier']
    top5_tiers = [df_enhanced.iloc[train_idx[j]]['urgency_tier']
                  for j in top5_j]
    top1_tier  = df_enhanced.iloc[train_idx[top1_j[0]]]['urgency_tier']
    if test_tier in top5_tiers: top5_match += 1
    if test_tier == top1_tier:  top1_match += 1

n_test = len(test_idx)
print(f"\n   Tier Match@1 (with HPO) : {top1_match/n_test:.2%}")
print(f"   Tier Match@5 (with HPO) : {top5_match/n_test:.2%}")

# ============================================
# SAVE ENHANCED MODEL
# ============================================
print("\nSaving enhanced model...")

model_data_enhanced = {
    'X_train'     : X_norm[train_idx],
    'df_train'    : df_enhanced.iloc[train_idx].reset_index(drop=True),
    'feature_cols': feature_cols_new
}

with open('output_model.pkl', 'wb') as f:
    pickle.dump(model_data_enhanced, f)
print("   OK output_model.pkl          (updated with HPO)")

with open('output_feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols_new, f)
print("   OK output_feature_cols.pkl   (updated with HPO)")

df_enhanced.to_csv('output_master_features.csv', index=False)
print("   OK output_master_features.csv (updated with HPO)")

# Save HPO term list for UI dropdowns
hpo_term_labels = hpo_filtered[[hpo_col]]\
    .drop_duplicates()\
    .rename(columns={hpo_col: 'hpo_id'})

# Try to get human-readable labels if available
hpo_term_labels.to_csv('output_hpo_terms.csv', index=False)
print("   OK output_hpo_terms.csv      (for UI dropdowns)")

print("\n" + "=" * 55)
print("  STEP 3b COMPLETE!")
print(f"  Model now uses {len(feature_cols_new):,} features")
print(f"  including {len(hpo_cols):,} HPO symptom terms.")
print("  Relaunch: streamlit run step4_doctor_interface.py")
print("=" * 55)