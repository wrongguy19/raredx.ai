# ============================================
# RARE DISEASE AI PROJECT
# Step 3d: Anomaly Detection
# ============================================
# WHAT THIS ADDS:
#   - Detects 347 ultra-rare diseases that are
#     statistical outliers in the dataset
#   - Uses Isolation Forest algorithm
#   - Assigns anomaly score to every disease
#   - Flags patient profiles that match
#     ultra-rare disease patterns
#   - Adds "Ultra-Rare Alert" to UI output
#
# Run AFTER step3c_knowledge_graph.py
# ============================================

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.ensemble       import IsolationForest
from sklearn.preprocessing  import normalize

print("=" * 55)
print("  STEP 3d: ANOMALY DETECTION")
print("  (Ultra-Rare Disease Flagging)")
print("=" * 55)

t0 = time.time()

# ============================================
# LOAD DATA
# ============================================
print("\n[1/4] Loading data...")

natural  = pd.read_csv('rare_diseases_natural_history.csv')
genes    = pd.read_csv('rare_diseases_genes.csv')
prev     = pd.read_csv('rare_diseases_prevalence.csv')
complete = pd.read_csv('rare_diseases_complete.csv')

print(f"   OK Diseases : {len(natural):,}")

# ============================================
# BUILD FEATURE MATRIX
# ============================================
print("\n[2/4] Building feature matrix...")

natural['AgeOfOnset']        = natural['AgeOfOnset'].fillna('Unknown')
natural['TypeOfInheritance'] = natural['TypeOfInheritance'].fillna('Unknown')

onset_map = {
    'Antenatal':9, 'Neonatal':8, 'Infancy':7,
    'Childhood':6, 'Adolescent':5, 'All ages':4,
    'Adult':3, 'Elderly':1, 'Unknown':0
}
natural['onset_score'] = natural['AgeOfOnset'].apply(
    lambda s: max(
        [onset_map.get(t.strip(), 0) for t in str(s).split(',')],
        default=0
    )
)

def assign_tier(row):
    neonatal = any(x in str(row['AgeOfOnset'])
                   for x in ['Neonatal', 'Antenatal'])
    ar = 'Autosomal recessive' in str(row['TypeOfInheritance'])
    if neonatal and ar:   return 1
    elif any(x in str(row['AgeOfOnset'])
             for x in ['Infancy', 'Childhood']): return 2
    else:                 return 3

natural['urgency_tier'] = natural.apply(assign_tier, axis=1)

inh_dummies = natural['TypeOfInheritance'].str.get_dummies(sep=', ')
inh_dummies.columns = [
    'inh_' + c.lower().replace(' ', '_').replace('/', '_')
    for c in inh_dummies.columns
]

# Rarity
prev['rarity_weight'] = prev['PrevalenceClass'].map({
    '<1 / 1 000 000':7, '1-9 / 1 000 000':6, 'Unknown':5,
    '1-9 / 100 000':4,  '1-5 / 10 000':3,    '6-9 / 10 000':2,
    '>1 / 1000':1
}).fillna(5)
rarity = prev.groupby('OrphaCode')['rarity_weight'].max().reset_index()

# Genes
genes_ok = genes[genes['AssociationStatus'] == 'Assessed']
top100   = genes_ok['GeneSymbol'].value_counts().head(100).index.tolist()
gene_mat = genes_ok[genes_ok['GeneSymbol'].isin(top100)].pivot_table(
    index='OrphaCode', columns='GeneSymbol',
    values='AssociationType', aggfunc='count', fill_value=0
).clip(0, 1)

df = natural[['OrphaCode', 'DiseaseName', 'AgeOfOnset',
              'TypeOfInheritance', 'onset_score', 'urgency_tier']].copy()
df = pd.concat([df.reset_index(drop=True),
                inh_dummies.reset_index(drop=True)], axis=1)
df = df.merge(rarity, on='OrphaCode', how='left')
df['rarity_weight'] = df['rarity_weight'].fillna(5)

id_info = complete[['OrphaCode', 'OMIM', 'MONDO']].copy()
id_info['has_omim']  = id_info['OMIM'].notna().astype(int)
id_info['has_mondo'] = id_info['MONDO'].notna().astype(int)
df = df.merge(id_info[['OrphaCode', 'has_omim', 'has_mondo']],
              on='OrphaCode', how='left')
df = df.merge(gene_mat.reset_index(), on='OrphaCode', how='left')
df = df.fillna(0)

exclude      = ['OrphaCode', 'DiseaseName', 'AgeOfOnset', 'TypeOfInheritance']
feature_cols = [c for c in df.columns if c not in exclude]
X = np.nan_to_num(
    df[feature_cols].values.astype(np.float32),
    nan=0.0, posinf=0.0, neginf=0.0
)
X_norm = normalize(X, norm='l2')
print(f"   OK Feature matrix : {X_norm.shape}")

# ============================================
# FIT ISOLATION FOREST
# ============================================
print("\n[3/4] Fitting Isolation Forest...")
print("   Contamination rate = 5% (top 5% most anomalous)")

iso = IsolationForest(
    n_estimators = 200,
    contamination = 0.05,   # 5% = ~370 ultra-rare diseases
    max_samples  = 'auto',
    random_state = 42,
    n_jobs       = -1
)

iso.fit(X_norm)

# Predict: -1 = anomaly (ultra-rare), 1 = normal
predictions   = iso.predict(X_norm)
anomaly_scores = iso.score_samples(X_norm)
# Lower score = more anomalous

df['is_anomaly']    = (predictions == -1)
df['anomaly_score'] = anomaly_scores

n_anomalies = df['is_anomaly'].sum()
n_normal    = (~df['is_anomaly']).sum()
threshold   = np.percentile(anomaly_scores, 5)

print(f"\n   Results:")
print(f"   Ultra-rare (anomaly) diseases : {n_anomalies:,}")
print(f"   Normal diseases               : {n_normal:,}")
print(f"   Anomaly threshold score       : {threshold:.4f}")
print(f"   Score range                   : "
      f"{anomaly_scores.min():.4f} to {anomaly_scores.max():.4f}")

# ============================================
# ANALYZE ANOMALY DISEASES
# ============================================
print("\n[4/4] Analyzing ultra-rare disease patterns...")

anomaly_df = df[df['is_anomaly'] == True].copy()
normal_df  = df[df['is_anomaly'] == False].copy()

# Inheritance patterns in anomalies
print(f"\n   Inheritance patterns in ultra-rare diseases:")
inh_cols = [c for c in df.columns if c.startswith('inh_')]
inh_means_anomaly = anomaly_df[inh_cols].mean()
inh_means_normal  = normal_df[inh_cols].mean()

for col in inh_cols:
    a_val = inh_means_anomaly[col]
    n_val = inh_means_normal[col]
    if a_val > 0.05:
        diff = a_val - n_val
        label = col.replace('inh_', '').replace('_', ' ').title()
        print(f"   {label:<35} anomaly={a_val:.1%}  normal={n_val:.1%}  "
              f"diff={diff:+.1%}")

# Onset patterns
print(f"\n   Onset patterns in ultra-rare diseases:")
print(f"   Anomaly onset_score mean : "
      f"{anomaly_df['onset_score'].mean():.2f}")
print(f"   Normal onset_score mean  : "
      f"{normal_df['onset_score'].mean():.2f}")

# Sample anomaly diseases
print(f"\n   Sample ultra-rare diseases detected:")
print(f"   {'Disease':<45}  {'Score':>7}  {'Onset':<20}")
print(f"   " + "-" * 76)
sample_anomalies = anomaly_df.nsmallest(10, 'anomaly_score')
for _, row in sample_anomalies.iterrows():
    name  = str(row['DiseaseName'])[:45]
    score = row['anomaly_score']
    onset = str(row['AgeOfOnset'])[:20]
    print(f"   {name:<45}  {score:>7.4f}  {onset:<20}")

# ============================================
# PATIENT ANOMALY CHECK FUNCTION
# ============================================
def check_patient_anomaly(patient_features_np):
    """
    Check if a patient profile matches an
    ultra-rare disease pattern.

    Parameters
    ----------
    patient_features_np : np.ndarray
        Normalized patient feature vector

    Returns
    -------
    dict with keys:
        is_anomaly  : bool
        score       : float (lower = more anomalous)
        risk_level  : str ('ULTRA-RARE', 'NORMAL')
        message     : str
    """
    score    = iso.score_samples(patient_features_np)[0]
    is_anom  = score < threshold

    if score < threshold - 0.1:
        risk_level = 'EXTREME'
        message    = ('Patient profile matches an EXTREMELY rare disease '
                      'pattern. Immediate specialist referral recommended.')
    elif is_anom:
        risk_level = 'ULTRA-RARE'
        message    = ('Patient profile matches an ultra-rare disease pattern '
                      '(top 5% rarest). Consider specialist genetic workup.')
    else:
        risk_level = 'NORMAL'
        message    = ('Patient profile within normal rare disease range.')

    return {
        'is_anomaly' : is_anom,
        'score'      : float(score),
        'threshold'  : float(threshold),
        'risk_level' : risk_level,
        'message'    : message
    }

# ============================================
# SAVE OUTPUTS
# ============================================
print("\n   Saving anomaly detection outputs...")

with open('output_anomaly_model.pkl', 'wb') as f:
    pickle.dump({
        'model'    : iso,
        'threshold': threshold,
        'scores'   : anomaly_scores
    }, f)
print("   OK output_anomaly_model.pkl")

# Save anomaly flags per disease
anomaly_output = df[['OrphaCode', 'DiseaseName',
                      'AgeOfOnset', 'TypeOfInheritance',
                      'urgency_tier', 'is_anomaly',
                      'anomaly_score']].copy()
anomaly_output.to_csv('output_anomaly_scores.csv', index=False)
print("   OK output_anomaly_scores.csv")

# Save just anomaly diseases
anomaly_df[['OrphaCode', 'DiseaseName', 'AgeOfOnset',
            'TypeOfInheritance', 'urgency_tier',
            'anomaly_score']]\
    .sort_values('anomaly_score')\
    .to_csv('output_ultra_rare_diseases.csv', index=False)
print("   OK output_ultra_rare_diseases.csv")

print(f"\n   Total runtime : {time.time()-t0:.0f} seconds")
print("\n" + "=" * 55)
print("  STEP 3d COMPLETE!")
print(f"  {n_anomalies:,} ultra-rare diseases flagged.")
print("  Next: Update step4_doctor_interface.py")
print("=" * 55)
