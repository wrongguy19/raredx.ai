# ============================================
# RARE DISEASE AI PROJECT
# Step 2: Feature Engineering
# ============================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 50)

# --- Load cleaned outputs from Step 1 ---
print("\nLoading Step 1 outputs...")
natural    = pd.read_csv('output_natural_with_tiers.csv')
genes      = pd.read_csv('output_genes_assessed.csv')
prevalence = pd.read_csv('rare_diseases_prevalence.csv')
complete   = pd.read_csv('rare_diseases_complete.csv')

print(f"✅ Natural history loaded  : {len(natural):,} records")
print(f"✅ Genes loaded            : {len(genes):,} records")
print(f"✅ Prevalence loaded       : {len(prevalence):,} records")

# ============================================
# FEATURE 1: Age of Onset → Urgency Score
# ============================================
print("\n[1/4] Encoding Age of Onset as urgency score...")

onset_map = {
    'Antenatal'  : 9,
    'Neonatal'   : 8,
    'Infancy'    : 7,
    'Childhood'  : 6,
    'Adolescent' : 5,
    'All ages'   : 4,
    'Adult'      : 3,
    'Elderly'    : 1,
    'No data available': 0
}

def encode_onset(onset_str):
    if pd.isna(onset_str):
        return 0
    scores = [
        onset_map.get(term.strip(), 0)
        for term in str(onset_str).split(',')
    ]
    return max(scores)  # Take highest urgency if multiple

natural['onset_score'] = natural['AgeOfOnset'].apply(encode_onset)

print(f"✅ Onset scores assigned")
print(f"   Score distribution:\n{natural['onset_score'].value_counts().sort_index().to_string()}")

# ============================================
# FEATURE 2: Inheritance Pattern → Binary Columns
# ============================================
print("\n[2/4] One-hot encoding inheritance patterns...")

inh_dummies = natural['TypeOfInheritance']\
    .fillna('Unknown')\
    .str.get_dummies(sep=', ')

# Clean column names
inh_dummies.columns = [
    'inh_' + c.lower().replace(' ', '_').replace('/', '_')
    for c in inh_dummies.columns
]

inh_dummies['OrphaCode'] = natural['OrphaCode'].values

print(f"✅ Inheritance features created : {len(inh_dummies.columns) - 1} columns")
print(f"   Features: {list(inh_dummies.columns[:-1])}")

# ============================================
# FEATURE 3: Gene Presence/Absence Matrix
# ============================================
print("\n[3/4] Building gene presence/absence matrix...")
print("      (This may take 30-60 seconds...)")

# Confidence weights by association type
assoc_weight_map = {
    'Disease-causing germline mutation(s) in'                      : 1.0,
    'Disease-causing germline mutation(s) (loss of function) in'   : 1.0,
    'Disease-causing germline mutation(s) (gain of function) in'   : 1.0,
    'Disease-causing somatic mutation(s) in'                       : 0.9,
    'Role in the phenotype of'                                      : 0.7,
    'Major susceptibility factor in'                               : 0.6,
    'Modifying germline mutation in'                               : 0.5,
    'Part of a fusion gene in'                                     : 0.5,
    'Biomarker tested in'                                          : 0.3,
    'Candidate gene tested in'                                     : 0.2,
}

genes['confidence_weight'] = genes['AssociationType'].map(assoc_weight_map).fillna(0.5)

# Build pivot matrix (diseases × genes)
gene_matrix = genes.pivot_table(
    index='OrphaCode',
    columns='GeneSymbol',
    values='confidence_weight',
    aggfunc='max',
    fill_value=0
)

print(f"✅ Gene matrix built")
print(f"   Shape : {gene_matrix.shape[0]:,} diseases × {gene_matrix.shape[1]:,} genes")
print(f"   Top 10 most common genes:")
gene_freq = (gene_matrix > 0).sum().sort_values(ascending=False).head(10)
for gene, count in gene_freq.items():
    print(f"   → {gene}: {count} diseases")

# ============================================
# FEATURE 4: Prevalence Rarity Weight
# ============================================
print("\n[4/4] Adding prevalence rarity weights...")

rarity_map = {
    '<1 / 1 000 000'    : 7,   # extremely rare
    '1-9 / 1 000 000'   : 6,
    'Unknown'           : 5,   # treat unknown as rare
    '1-9 / 100 000'     : 4,
    '1-5 / 10 000'      : 3,
    '6-9 / 10 000'      : 2,
    '>1 / 1000'         : 1,   # least rare
    'Not yet documented': 5
}

prevalence['rarity_weight'] = prevalence['PrevalenceClass'].map(rarity_map).fillna(5)

# Keep one rarity score per disease (take max = most severe estimate)
rarity_per_disease = prevalence.groupby('OrphaCode')['rarity_weight'].max().reset_index()
rarity_per_disease.columns = ['OrphaCode', 'rarity_weight']

print(f"✅ Rarity weights assigned to {len(rarity_per_disease):,} diseases")

# ============================================
# MERGE ALL FEATURES INTO MASTER TABLE
# ============================================
print("\nMerging all features into master table...")

# Start with natural history base
base = natural[['OrphaCode', 'DiseaseName', 'AgeOfOnset',
                 'TypeOfInheritance', 'urgency_tier', 'onset_score']].copy()

# Add inheritance dummies
base = base.merge(inh_dummies, on='OrphaCode', how='left')

# Add rarity weight
base = base.merge(rarity_per_disease, on='OrphaCode', how='left')
base['rarity_weight'] = base['rarity_weight'].fillna(5)

# Add OMIM & MONDO from complete (needed for HPO linkage in Step 3)
id_cols = complete[['OrphaCode', 'OMIM', 'MONDO', 'ICD-10', 'DisorderType']]
base = base.merge(id_cols, on='OrphaCode', how='left')

# Add gene matrix
base = base.merge(gene_matrix.reset_index(), on='OrphaCode', how='left')
gene_cols = gene_matrix.columns.tolist()
base[gene_cols] = base[gene_cols].fillna(0)

print(f"✅ Master feature table built!")
print(f"   Total records  : {len(base):,}")
print(f"   Total features : {len(base.columns):,}")
print(f"\n   Feature breakdown:")
print(f"   → Onset score          : 1 column")
print(f"   → Inheritance patterns : {len(inh_dummies.columns) - 1} columns")
print(f"   → Gene presence matrix : {len(gene_cols):,} columns")
print(f"   → Rarity weight        : 1 column")
print(f"   → ID columns (OMIM etc): 4 columns")

# ============================================
# QUICK SANITY CHECK
# ============================================
print("\nRunning sanity checks...")

tier1 = base[base['urgency_tier'] == 1]
tier2 = base[base['urgency_tier'] == 2]
has_genes = base[(base[gene_cols] > 0).any(axis=1)]

print(f"✅ Tier 1 (Critical) diseases  : {len(tier1):,}")
print(f"✅ Tier 2 (Urgent) diseases    : {len(tier2):,}")
print(f"✅ Diseases with gene data     : {len(has_genes):,}")
print(f"✅ OMIM IDs available          : {base['OMIM'].notna().sum():,}")
print(f"✅ MONDO IDs available         : {base['MONDO'].notna().sum():,}")

# ============================================
# SAVE OUTPUTS
# ============================================
print("\nSaving outputs...")

# Save full master table (all features)
base.to_csv('output_master_features.csv', index=False)

# Save a lightweight version (no gene columns) for quick inspection
light_cols = ['OrphaCode', 'DiseaseName', 'AgeOfOnset', 'TypeOfInheritance',
              'urgency_tier', 'onset_score', 'rarity_weight',
              'OMIM', 'MONDO', 'ICD-10', 'DisorderType']
base[light_cols].to_csv('output_features_lightweight.csv', index=False)

# Save gene matrix separately
gene_matrix.to_csv('output_gene_matrix.csv')

print("✅ output_master_features.csv     → Full feature table (for ML model)")
print("✅ output_features_lightweight.csv → Summary table (for quick inspection)")
print("✅ output_gene_matrix.csv          → Gene presence matrix")

print("\n" + "=" * 50)
print("🎉 Step 2 Complete!")
print("   Your data is now ML-ready.")
print("   Next: Run step3_model_training.py")
print("=" * 50)