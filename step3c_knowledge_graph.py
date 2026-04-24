# ============================================
# RARE DISEASE AI PROJECT
# Step 3c: Gene-Disease Knowledge Graph
# ============================================
# WHAT THIS ADDS:
#   - Builds a bipartite graph:
#     Disease nodes + Gene nodes + edges
#   - Builds a disease-disease graph:
#     Two diseases connected if they share a gene
#   - Finds hub diseases (most connected)
#   - Enables graph-based recommendations:
#     "Also consider these related diseases"
#   - Computes graph metrics for each disease
#
# Run AFTER step3b_clustering.py
# ============================================

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
import warnings
import time
import json
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    print("WARNING: networkx not installed.")
    print("Run: pip install networkx")
    print("Continuing with manual graph implementation...")

print("=" * 55)
print("  STEP 3c: KNOWLEDGE GRAPH BUILDER")
print("=" * 55)

t0 = time.time()

# ============================================
# LOAD DATA
# ============================================
print("\n[1/5] Loading data...")

natural  = pd.read_csv('rare_diseases_natural_history.csv')
genes    = pd.read_csv('rare_diseases_genes.csv')
complete = pd.read_csv('rare_diseases_complete.csv')

genes_ok = genes[genes['AssociationStatus'] == 'Assessed'].copy()

assoc_weight = {
    'Disease-causing germline mutation(s) in'                    : 1.0,
    'Disease-causing germline mutation(s) (loss of function) in' : 1.0,
    'Disease-causing germline mutation(s) (gain of function) in' : 1.0,
    'Disease-causing somatic mutation(s) in'                     : 0.9,
    'Role in the phenotype of'                                   : 0.7,
    'Major susceptibility factor in'                             : 0.6,
    'Modifying germline mutation in'                             : 0.5,
    'Part of a fusion gene in'                                   : 0.5,
    'Biomarker tested in'                                        : 0.3,
    'Candidate gene tested in'                                   : 0.2,
}
genes_ok['weight'] = genes_ok['AssociationType'].map(assoc_weight).fillna(0.5)

print(f"   OK Diseases : {natural['OrphaCode'].nunique():,}")
print(f"   OK Genes    : {genes_ok['GeneSymbol'].nunique():,}")

# ============================================
# BUILD BIPARTITE GRAPH (Disease + Gene)
# ============================================
print("\n[2/5] Building bipartite disease-gene graph...")

disease_info = natural.set_index('OrphaCode')[['DiseaseName','AgeOfOnset','TypeOfInheritance']].to_dict('index')

if NX_AVAILABLE:
    G = nx.Graph()

    # Add disease nodes
    for orpha, info in disease_info.items():
        G.add_node(
            int(orpha),
            node_type = 'disease',
            name      = str(info['DiseaseName']),
            onset     = str(info['AgeOfOnset']),
            inh       = str(info['TypeOfInheritance'])
        )

    # Add gene nodes and edges
    for _, row in genes_ok.iterrows():
        gene  = str(row['GeneSymbol'])
        orpha = int(row['OrphaCode'])
        if gene not in G.nodes:
            G.add_node(gene, node_type='gene')
        G.add_edge(orpha, gene, weight=row['weight'],
                   assoc_type=str(row['AssociationType']))

    print(f"   OK Bipartite nodes  : {G.number_of_nodes():,}")
    print(f"   OK Bipartite edges  : {G.number_of_edges():,}")
    print(f"   OK Disease nodes    : "
          f"{sum(1 for n,d in G.nodes(data=True) if d.get('node_type')=='disease'):,}")
    print(f"   OK Gene nodes       : "
          f"{sum(1 for n,d in G.nodes(data=True) if d.get('node_type')=='gene'):,}")
else:
    G = None
    print("   OK Skipped (networkx not available)")

# ============================================
# BUILD DISEASE-DISEASE GRAPH (Shared Genes)
# ============================================
print("\n[3/5] Building disease-disease graph (shared genes)...")

# Map gene -> list of diseases
gene_to_diseases = (
    genes_ok.groupby('GeneSymbol')['OrphaCode']
    .apply(list)
    .to_dict()
)

# Build adjacency manually (works without networkx)
disease_adjacency = {}   # orpha -> {orpha: [genes]}
shared_gene_count  = 0

for gene, disease_list in gene_to_diseases.items():
    if len(disease_list) < 2:
        continue
    for i in range(len(disease_list)):
        for j in range(i + 1, len(disease_list)):
            d1 = int(disease_list[i])
            d2 = int(disease_list[j])
            if d1 not in disease_adjacency:
                disease_adjacency[d1] = {}
            if d2 not in disease_adjacency:
                disease_adjacency[d2] = {}
            if d2 not in disease_adjacency[d1]:
                disease_adjacency[d1][d2] = []
                disease_adjacency[d2][d1] = []
                shared_gene_count += 1
            disease_adjacency[d1][d2].append(gene)
            disease_adjacency[d2][d1].append(gene)

total_edges     = sum(len(v) for v in disease_adjacency.values()) // 2
connected_nodes = len(disease_adjacency)

print(f"   OK Disease-Disease edges     : {total_edges:,}")
print(f"   OK Connected disease nodes   : {connected_nodes:,}")
print(f"   OK Isolated diseases         : "
      f"{natural['OrphaCode'].nunique() - connected_nodes:,}")

# ============================================
# FIND HUB DISEASES (Most Connected)
# ============================================
print("\n[4/5] Computing graph metrics...")

# Degree (number of disease connections)
degree_map = {
    d: len(neighbours)
    for d, neighbours in disease_adjacency.items()
}

# Top hub diseases
top_hubs = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\n   Top 10 hub diseases (most gene connections):")
print(f"   {'Disease':<45} {'Connections':>11}")
print(f"   " + "-" * 58)
for orpha, degree in top_hubs:
    info  = disease_info.get(int(orpha), {})
    name  = str(info.get('DiseaseName', f'OrphaCode:{orpha}'))[:45]
    print(f"   {name:<45} {degree:>11,}")

# Shared gene strength
print(f"\n   Top 10 strongest disease pairs (most shared genes):")
print(f"   {'Disease Pair':<50} {'Shared Genes':>12}")
print(f"   " + "-" * 64)
pairs = []
for d1, neighbours in disease_adjacency.items():
    for d2, shared in neighbours.items():
        if d1 < d2:
            pairs.append((d1, d2, len(shared), shared))
pairs.sort(key=lambda x: x[2], reverse=True)
for d1, d2, n_shared, shared_genes in pairs[:10]:
    name1 = str(disease_info.get(d1, {}).get('DiseaseName', d1))[:22]
    name2 = str(disease_info.get(d2, {}).get('DiseaseName', d2))[:22]
    pair_str = f"{name1} + {name2}"[:50]
    print(f"   {pair_str:<50} {n_shared:>12}  ({', '.join(shared_genes[:3])})")

# ============================================
# GRAPH-ENHANCED RECOMMENDATION FUNCTION
# ============================================
def get_graph_neighbours(orpha_code, top_n=3):
    """
    Given a disease OrphaCode, return
    the most related diseases via shared genes.

    Parameters
    ----------
    orpha_code : int
        The OrphaCode of the disease
    top_n : int
        Number of neighbours to return

    Returns
    -------
    list of dict with keys:
        orpha_code, name, shared_genes, n_shared
    """
    orpha_code = int(orpha_code)
    if orpha_code not in disease_adjacency:
        return []

    neighbours = disease_adjacency[orpha_code]
    sorted_nb  = sorted(
        neighbours.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:top_n]

    results = []
    for nb_code, shared in sorted_nb:
        info = disease_info.get(int(nb_code), {})
        results.append({
            'orpha_code'  : int(nb_code),
            'name'        : str(info.get('DiseaseName', f'OrphaCode:{nb_code}')),
            'shared_genes': shared[:5],    # top 5 shared genes
            'n_shared'    : len(shared),
            'onset'       : str(info.get('AgeOfOnset', 'Unknown')),
            'inheritance' : str(info.get('TypeOfInheritance', 'Unknown'))
        })
    return results


# Test the function
print(f"\n   Test: Graph neighbours of 'Pompe disease' (OrphaCode 365):")
test_result = get_graph_neighbours(365, top_n=3)
if test_result:
    for r in test_result:
        print(f"     -> {r['name'][:45]} "
              f"(shared: {', '.join(r['shared_genes'][:3])})")
else:
    print("     -> No graph connections for this disease")

# ============================================
# COMPUTE GRAPH METRICS PER DISEASE
# ============================================
print("\n   Computing connectivity metrics for all diseases...")

graph_metrics = {}
for orpha in natural['OrphaCode'].tolist():
    orpha = int(orpha)
    degree  = degree_map.get(orpha, 0)
    # Sum of shared gene counts = weighted degree
    w_degree = sum(
        len(genes_list)
        for genes_list in disease_adjacency.get(orpha, {}).values()
    )
    graph_metrics[orpha] = {
        'degree'  : degree,
        'w_degree': w_degree,
        'is_hub'  : degree >= 20
    }

hub_count = sum(1 for m in graph_metrics.values() if m['is_hub'])
print(f"   OK Hub diseases (20+ connections)  : {hub_count:,}")
print(f"   OK Connected diseases              : {connected_nodes:,}")
print(f"   OK Isolated diseases               : "
      f"{natural['OrphaCode'].nunique() - connected_nodes:,}")

# ============================================
# SAVE OUTPUTS
# ============================================
print("\n   Saving graph outputs...")

# Save adjacency dict
with open('output_disease_graph.pkl', 'wb') as f:
    pickle.dump(disease_adjacency, f)
print("   OK output_disease_graph.pkl")

# Save graph metrics
with open('output_graph_metrics.pkl', 'wb') as f:
    pickle.dump(graph_metrics, f)
print("   OK output_graph_metrics.pkl")

# Save disease info dict
with open('output_disease_info.pkl', 'wb') as f:
    pickle.dump(disease_info, f)
print("   OK output_disease_info.pkl")

# Save top pairs as CSV
pairs_df = pd.DataFrame([
    {
        'disease1_code'   : d1,
        'disease1_name'   : disease_info.get(d1, {}).get('DiseaseName', ''),
        'disease2_code'   : d2,
        'disease2_name'   : disease_info.get(d2, {}).get('DiseaseName', ''),
        'n_shared_genes'  : n_shared,
        'shared_genes'    : ', '.join(shared[:5])
    }
    for d1, d2, n_shared, shared in pairs[:100]
])
pairs_df.to_csv('output_top_gene_pairs.csv', index=False)
print("   OK output_top_gene_pairs.csv")

# Save hub diseases
hub_df = pd.DataFrame([
    {
        'OrphaCode'  : orpha,
        'DiseaseName': disease_info.get(orpha, {}).get('DiseaseName', ''),
        'connections': degree_map.get(orpha, 0)
    }
    for orpha in [h[0] for h in top_hubs]
])
hub_df.to_csv('output_hub_diseases.csv', index=False)
print("   OK output_hub_diseases.csv")

print(f"\n   Total runtime : {time.time()-t0:.0f} seconds")
print("\n" + "=" * 55)
print("  STEP 3c COMPLETE!")
print(f"  Graph built: {total_edges:,} disease connections")
print(f"  {hub_count:,} hub diseases identified")
print("  Next: Run step3d_anomaly.py")
print("=" * 55)
