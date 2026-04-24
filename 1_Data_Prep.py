# ============================================
# RARE DISEASE AI PROJECT
# Step 1: Data Loading & Preparation
# ============================================

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")

# --- Load all 5 files ---
complete  = pd.read_csv('rare_diseases_complete.csv')
genes     = pd.read_csv('rare_diseases_genes.csv')
info      = pd.read_csv('rare_diseases_info.csv')
natural   = pd.read_csv('rare_diseases_natural_history.csv')
prevalence= pd.read_csv('rare_diseases_prevalence.csv')

print(f"✅ Complete dataset    : {len(complete):,} records")
print(f"✅ Genes dataset       : {len(genes):,} records")
print(f"✅ Info dataset        : {len(info):,} records")
print(f"✅ Natural history     : {len(natural):,} records")
print(f"✅ Prevalence dataset  : {len(prevalence):,} records")

# --- Filter to Pediatric Diseases ---
print("\nFiltering pediatric-onset diseases...")

pediatric_keywords = 'Neonatal|Infancy|Childhood|Antenatal'
pediatric = natural[
    natural['AgeOfOnset'].str.contains(pediatric_keywords, na=False)
]
print(f"✅ Pediatric diseases found : {pediatric['OrphaCode'].nunique():,}")

# --- Assign Urgency Tiers ---
print("\nAssigning urgency tiers...")

def assign_tier(row):
    neonatal = any(x in str(row['AgeOfOnset'])
                   for x in ['Neonatal', 'Antenatal'])
    ar = 'Autosomal recessive' in str(row['TypeOfInheritance'])
    if neonatal and ar:
        return 1  # 🔴 Most Critical
    elif any(x in str(row['AgeOfOnset'])
             for x in ['Infancy', 'Childhood']):
        return 2  # 🟡 Urgent
    else:
        return 3  # 🟢 Monitor

natural['urgency_tier'] = natural.apply(assign_tier, axis=1)
tier_counts = natural['urgency_tier'].value_counts().sort_index()

print(f"🔴 Tier 1 (Critical - Neonatal AR) : {tier_counts.get(1, 0):,} diseases")
print(f"🟡 Tier 2 (Urgent - Childhood)     : {tier_counts.get(2, 0):,} diseases")
print(f"🟢 Tier 3 (Monitor - Later onset)  : {tier_counts.get(3, 0):,} diseases")

# --- Filter genes to Assessed only ---
print("\nFiltering confirmed gene associations...")
genes_assessed = genes[genes['AssociationStatus'] == 'Assessed']
print(f"✅ Assessed gene records : {len(genes_assessed):,}")
print(f"✅ Unique genes          : {genes_assessed['GeneSymbol'].nunique():,}")

# --- Merge pediatric + genes ---
ped_with_genes = pediatric.merge(genes_assessed, on='OrphaCode')
print(f"✅ Pediatric diseases WITH gene data : {ped_with_genes['OrphaCode'].nunique():,}")

# --- Save cleaned outputs ---
print("\nSaving cleaned files...")
pediatric.to_csv('output_pediatric_diseases.csv', index=False)
genes_assessed.to_csv('output_genes_assessed.csv', index=False)
natural.to_csv('output_natural_with_tiers.csv', index=False)

print("\n🎉 Step 1 Complete! 3 output files saved to your project folder.")
print("Next: Run step2_feature_engineering.py")