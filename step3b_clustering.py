# ============================================
# RARE DISEASE AI PROJECT
# Step 3b: Disease Clustering
# ============================================
# WHAT THIS ADDS:
#   - Groups 7,374 diseases into 15 clusters
#     based on onset, inheritance, genes, rarity
#   - Assigns a clinical label to each cluster
#   - Enables hierarchical retrieval:
#     Patient → Cluster → Top diseases
#   - Saves cluster model for use in UI
#
# Run AFTER step3_model_training.py
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

from sklearn.cluster        import KMeans
from sklearn.preprocessing  import normalize
from sklearn.metrics        import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 55)
print("  STEP 3b: DISEASE CLUSTERING")
print("=" * 55)

t0 = time.time()

# ============================================
# LOAD RAW FILES
# ============================================
print("\n[1/5] Loading raw data files...")

natural  = pd.read_csv('rare_diseases_natural_history.csv')
genes    = pd.read_csv('rare_diseases_genes.csv')
prev     = pd.read_csv('rare_diseases_prevalence.csv')
complete = pd.read_csv('rare_diseases_complete.csv')

print(f"   OK Natural history : {len(natural):,} records")
print(f"   OK Genes           : {len(genes):,} records")

# ============================================
# BUILD FEATURE MATRIX
# (Same as step3 for consistency)
# ============================================
print("\n[2/5] Building feature matrix...")

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

genes_ok = genes[genes['AssociationStatus'] == 'Assessed']
top100   = genes_ok['GeneSymbol'].value_counts().head(100).index.tolist()
gene_mat = genes_ok[genes_ok['GeneSymbol'].isin(top100)].pivot_table(
    index='OrphaCode', columns='GeneSymbol',
    values='AssociationType', aggfunc='count', fill_value=0
).clip(0, 1)

prev['rarity_weight'] = prev['PrevalenceClass'].map({
    '<1 / 1 000 000':7, '1-9 / 1 000 000':6, 'Unknown':5,
    '1-9 / 100 000':4,  '1-5 / 10 000':3,    '6-9 / 10 000':2,
    '>1 / 1000':1
}).fillna(5)
rarity = prev.groupby('OrphaCode')['rarity_weight'].max().reset_index()

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
# FIND OPTIMAL CLUSTERS
# ============================================
print("\n[3/5] Finding optimal number of clusters...")

scores = {}
for k in [4, 6, 8, 10, 12, 15]:
    km = KMeans(n_clusters=k, random_state=42,
                n_init=10, max_iter=300)
    labels = km.fit_predict(X_norm)
    score  = silhouette_score(
        X_norm, labels, sample_size=2000, random_state=42
    )
    scores[k] = score
    print(f"   k={k:2d} : silhouette = {score:.4f}")

best_k = max(scores, key=scores.get)
print(f"\n   Best k = {best_k}  (silhouette = {scores[best_k]:.4f})")

# ============================================
# FINAL CLUSTERING
# ============================================
print(f"\n[4/5] Fitting final KMeans (k={best_k})...")

km_final   = KMeans(n_clusters=best_k, random_state=42,
                    n_init=10, max_iter=300)
df['cluster'] = km_final.fit_predict(X_norm)

# ============================================
# CHARACTERIZE & NAME CLUSTERS
# ============================================
print("\n   Cluster Summary:")
print(f"   {'ID':>3}  {'Size':>5}  {'Tier1%':>6}  "
      f"{'Dominant Onset':<25}  {'Dominant Inheritance':<30}")
print("   " + "-" * 80)

cluster_info = {}
for c in range(best_k):
    cdf     = df[df['cluster'] == c]
    onset   = (cdf['AgeOfOnset'].mode()[0]
               if len(cdf) > 0 else 'Unknown')
    inh     = (cdf['TypeOfInheritance'].mode()[0]
               if len(cdf) > 0 else 'Unknown')
    tier1   = (cdf['urgency_tier'] == 1).mean() * 100
    rarity_mean = cdf['rarity_weight'].mean()

    # Auto-generate clinical label
    onset_short = str(onset).split(',')[0].strip()
    inh_short   = str(inh).split(',')[0].strip()
    if tier1 > 50:
        label = f"Critical Neonatal ({inh_short})"
    elif 'Childhood' in str(onset):
        label = f"Childhood Onset ({inh_short})"
    elif 'Adult' in str(onset):
        label = f"Adult Onset ({inh_short})"
    elif 'All ages' in str(onset):
        label = f"All Ages ({inh_short})"
    else:
        label = f"{onset_short} ({inh_short})"

    cluster_info[c] = {
        'cluster_id'    : c,
        'label'         : label,
        'size'          : len(cdf),
        'tier1_pct'     : round(tier1, 1),
        'dominant_onset': str(onset)[:40],
        'dominant_inh'  : str(inh)[:40],
        'rarity_mean'   : round(rarity_mean, 2),
        'orpha_codes'   : cdf['OrphaCode'].tolist()
    }

    print(f"   {c:>3}  {len(cdf):>5}  {tier1:>5.1f}%  "
          f"{str(onset)[:25]:<25}  {str(inh)[:30]:<30}")

# ============================================
# EVALUATE: HIERARCHICAL vs FLAT
# ============================================
print("\n[5/5] Comparing hierarchical vs flat retrieval...")

rng       = np.random.RandomState(42)
idx       = rng.permutation(len(df))
train_idx = idx[:int(0.8 * len(df))]
test_idx  = idx[int(0.8 * len(df)):]

X_train  = X_norm[train_idx]
X_test   = X_norm[test_idx]
df_train = df.iloc[train_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)

# Flat similarity
sim_flat   = cosine_similarity(X_test, X_train)
flat_top1  = sum(
    1 for i in range(len(test_idx))
    if df_test.iloc[i]['urgency_tier'] ==
    df_train.iloc[sim_flat[i].argsort()[-1]]['urgency_tier']
)
flat_top5  = sum(
    1 for i in range(len(test_idx))
    if df_test.iloc[i]['urgency_tier'] in [
        df_train.iloc[j]['urgency_tier']
        for j in sim_flat[i].argsort()[-5:]
    ]
)

# Hierarchical similarity
test_clusters = km_final.predict(X_test)
hier_top1 = 0
hier_top5 = 0
for i in range(len(test_idx)):
    c    = test_clusters[i]
    mask = (df_train['cluster'] == c).values
    if mask.sum() == 0:
        mask = np.ones(len(df_train), dtype=bool)
    sims  = cosine_similarity(X_test[i:i+1], X_train[mask])[0]
    df_c  = df_train[mask].reset_index(drop=True)
    top5j = sims.argsort()[-5:][::-1]
    top1j = sims.argsort()[-1:][::-1]
    test_tier = df_test.iloc[i]['urgency_tier']
    if test_tier == df_c.iloc[top1j[0]]['urgency_tier']: hier_top1 += 1
    if test_tier in [df_c.iloc[j]['urgency_tier'] for j in top5j]: hier_top5 += 1

n_test = len(test_idx)
print(f"\n   Method              Match@1    Match@5")
print(f"   " + "-" * 40)
print(f"   Flat Cosine         {flat_top1/n_test:.2%}     {flat_top5/n_test:.2%}")
print(f"   Hierarchical        {hier_top1/n_test:.2%}     {hier_top5/n_test:.2%}")
print(f"   Improvement      {(hier_top1-flat_top1)/n_test:+.2%}      {(hier_top5-flat_top5)/n_test:+.2%}")

# ============================================
# SAVE OUTPUTS
# ============================================
print("\n   Saving clustering outputs...")

with open('output_cluster_model.pkl', 'wb') as f:
    pickle.dump(km_final, f)
print("   OK output_cluster_model.pkl")

with open('output_cluster_info.pkl', 'wb') as f:
    pickle.dump(cluster_info, f)
print("   OK output_cluster_info.pkl")

# Save cluster assignments
df[['OrphaCode', 'DiseaseName', 'AgeOfOnset',
    'TypeOfInheritance', 'urgency_tier', 'cluster']]\
    .to_csv('output_cluster_assignments.csv', index=False)
print("   OK output_cluster_assignments.csv")

# Save cluster summary for UI
cluster_summary = pd.DataFrame([
    {
        'cluster_id'    : v['cluster_id'],
        'label'         : v['label'],
        'size'          : v['size'],
        'tier1_pct'     : v['tier1_pct'],
        'dominant_onset': v['dominant_onset'],
        'dominant_inh'  : v['dominant_inh'],
    }
    for v in cluster_info.values()
])
cluster_summary.to_csv('output_cluster_summary.csv', index=False)
print("   OK output_cluster_summary.csv")

print(f"\n   Total runtime : {time.time()-t0:.0f} seconds")
print("\n" + "=" * 55)
print("  STEP 3b COMPLETE!")
print(f"  {best_k} disease clusters identified and saved.")
print("  Next: Run step3c_knowledge_graph.py")
print("=" * 55)
