# ============================================
# RARE DISEASE AI PROJECT
# Step 4: Doctor Interface v2.0
# ============================================
# NEW IN THIS VERSION:
#   Layer 1 : Cosine Similarity (base)
#   Layer 2 : Disease Clustering (15 groups)
#   Layer 3 : Knowledge Graph (gene links)
#   Layer 4 : Anomaly Detection (ultra-rare)
#   + HPO Symptom Selection
#
# Run: streamlit run step4_doctor_interface.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing    import normalize

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title = "RareDx AI v2.0",
    page_icon  = "🧬",
    layout     = "wide"
)

# ============================================
# STYLING
# ============================================
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    .header-box {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 100%);
        border: 1px solid #845EF7;
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 18px;
    }
    .header-title {
        font-size: 1.8rem; font-weight: 800;
        color: #F0F0FF; margin: 0 0 4px 0;
    }
    .header-sub { color: #8888AA; font-size: 0.85rem; margin: 0; }
    .header-badge {
        display: inline-block;
        background: rgba(132,94,247,0.15);
        border: 1px solid rgba(132,94,247,0.35);
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.72rem; color: #A78BFA;
        margin: 6px 4px 0 0;
    }

    .layer-badge {
        display: inline-block;
        border-radius: 4px; padding: 2px 8px;
        font-size: 0.7rem; font-weight: 700;
        margin-right: 6px;
    }
    .layer-1 { background:#845EF722; color:#845EF7; border:1px solid #845EF7; }
    .layer-2 { background:#00C9A722; color:#00C9A7; border:1px solid #00C9A7; }
    .layer-3 { background:#FFA50022; color:#FFA500; border:1px solid #FFA500; }
    .layer-4 { background:#FF444422; color:#FF4444; border:1px solid #FF4444; }

    .result-card {
        border-radius: 10px; padding: 14px 18px;
        margin-bottom: 10px; border-left: 5px solid;
    }
    .result-critical { background:rgba(255,68,68,0.08); border-color:#FF4444; }
    .result-urgent   { background:rgba(255,165,0,0.08); border-color:#FFA500; }
    .result-monitor  { background:rgba(0,201,167,0.08); border-color:#00C9A7; }

    .graph-card {
        background: rgba(255,165,0,0.06);
        border: 1px solid rgba(255,165,0,0.3);
        border-radius: 8px; padding: 10px 14px;
        margin-top: 10px;
    }
    .anomaly-banner {
        background: rgba(255,68,68,0.12);
        border: 2px solid #FF4444;
        border-radius: 10px; padding: 14px 18px;
        margin-bottom: 16px;
    }
    .cluster-badge {
        display: inline-block;
        background: rgba(0,201,167,0.12);
        border: 1px solid rgba(0,201,167,0.4);
        border-radius: 6px; padding: 4px 12px;
        font-size: 0.8rem; color: #00C9A7;
        margin-bottom: 12px;
    }
    .metric-box {
        background: #1A1A2E; border: 1px solid #2A2A4A;
        border-radius: 8px; padding: 10px 6px; text-align: center;
    }
    .metric-val { font-size: 1.3rem; font-weight: 700; color: #845EF7; }
    .metric-lbl { font-size: 0.62rem; color: #555580;
                  text-transform: uppercase; letter-spacing: 1px; }
    .info-badge {
        display: inline-block; background: #1A1A2E;
        border: 1px solid #333366; border-radius: 20px;
        padding: 2px 9px; font-size: 0.75rem;
        color: #8888CC; margin-right: 4px; margin-bottom: 3px;
    }
    .hpo-tag {
        display: inline-block;
        background: rgba(0,201,167,0.1);
        border: 1px solid rgba(0,201,167,0.3);
        border-radius: 4px; padding: 1px 7px;
        font-size: 0.7rem; color: #00C9A7; margin: 2px;
    }
    .divider { border:none; border-top:1px solid #1E1E3A; margin:12px 0; }
    .stButton > button {
        background: linear-gradient(135deg, #845EF7, #5C3D99) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; font-weight: 700 !important;
        width: 100% !important;
    }
    .disclaimer {
        background: rgba(255,165,0,0.07);
        border: 1px solid rgba(255,165,0,0.3);
        border-radius: 8px; padding: 9px 13px;
        color: #CC8800; font-size: 0.78rem; margin-top: 12px;
    }
    .empty-state {
        background: #12121F; border: 1px dashed #2A2A4A;
        border-radius: 12px; padding: 56px 32px; text-align: center;
    }
    .section-label {
        font-size: 0.7rem; font-weight: 700; color: #845EF7;
        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HPO SYMPTOM DICT
# ============================================
HPO_CATEGORIES = {
    "Neurological": {
        "HP:0001250": "Seizures",
        "HP:0001251": "Ataxia",
        "HP:0001257": "Spasticity",
        "HP:0001263": "Global developmental delay",
        "HP:0001290": "Hypotonia",
        "HP:0001324": "Muscle weakness",
        "HP:0001256": "Intellectual disability",
    },
    "Muscular": {
        "HP:0003326": "Myalgia (muscle pain)",
        "HP:0003560": "Muscular dystrophy",
        "HP:0003236": "Elevated CK levels",
    },
    "Visual": {
        "HP:0000572": "Visual loss",
        "HP:0000639": "Nystagmus",
        "HP:0000486": "Strabismus",
        "HP:0001087": "Retinal degeneration",
    },
    "Hearing": {
        "HP:0000365": "Hearing impairment",
        "HP:0000407": "Sensorineural hearing loss",
    },
    "Skeletal": {
        "HP:0002650": "Scoliosis",
        "HP:0001166": "Arachnodactyly",
        "HP:0004322": "Short stature",
        "HP:0001373": "Joint dislocation",
    },
    "Cardiac": {
        "HP:0001638": "Cardiomyopathy",
        "HP:0001644": "Dilated cardiomyopathy",
        "HP:0001631": "Atrial septal defect",
    },
    "Metabolic": {
        "HP:0001508": "Failure to thrive",
        "HP:0001943": "Hypoglycemia",
        "HP:0003128": "Lactic acidosis",
    },
    "Respiratory": {
        "HP:0002088": "Abnormal lung morphology",
        "HP:0002093": "Respiratory insufficiency",
    },
    "Renal": {
        "HP:0000077": "Abnormality of the kidney",
        "HP:0000107": "Renal cyst",
    },
}
HPO_FLAT = {k: v for cat in HPO_CATEGORIES.values() for k, v in cat.items()}

# ============================================
# LOAD ALL MODEL FILES
# ============================================
@st.cache_resource(show_spinner="Loading AI model layers...")
def load_all_models():
    errors = []
    out    = {}

    # Layer 1: Base similarity model (required)
    try:
        with open('output_model.pkl', 'rb') as f:
            out['model'] = pickle.load(f)
        with open('output_feature_cols.pkl', 'rb') as f:
            out['feature_cols'] = pickle.load(f)
        out['df_master'] = pd.read_csv('output_master_features.csv')
        X_raw = np.nan_to_num(
            out['df_master'][out['feature_cols']].values.astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        out['X_norm'] = normalize(X_raw, norm='l2')
        out['layer1'] = True
    except FileNotFoundError as e:
        errors.append(f"Layer 1 (model): {e}")
        out['layer1'] = False

    # Layer 2: Clustering (optional)
    try:
        with open('output_cluster_model.pkl', 'rb') as f:
            out['cluster_model'] = pickle.load(f)
        with open('output_cluster_info.pkl', 'rb') as f:
            out['cluster_info'] = pickle.load(f)
        out['cluster_assignments'] = pd.read_csv(
            'output_cluster_assignments.csv'
        )
        out['layer2'] = True
    except FileNotFoundError:
        out['layer2'] = False

    # Layer 3: Knowledge Graph (optional)
    try:
        with open('output_disease_graph.pkl', 'rb') as f:
            out['disease_graph'] = pickle.load(f)
        with open('output_disease_info.pkl', 'rb') as f:
            out['disease_info'] = pickle.load(f)
        out['layer3'] = True
    except FileNotFoundError:
        out['layer3'] = False

    # Layer 4: Anomaly detection (optional)
    try:
        with open('output_anomaly_model.pkl', 'rb') as f:
            anom = pickle.load(f)
            out['anomaly_model']    = anom['model']
            out['anomaly_threshold'] = anom['threshold']
        out['layer4'] = True
    except FileNotFoundError:
        out['layer4'] = False

    out['errors'] = errors
    return out

models = load_all_models()

# ============================================
# HEADER
# ============================================
l1 = "✅" if models.get('layer1') else "❌"
l2 = "✅" if models.get('layer2') else "⚠️ Run step3b"
l3 = "✅" if models.get('layer3') else "⚠️ Run step3c"
l4 = "✅" if models.get('layer4') else "⚠️ Run step3d"
n_diseases = len(models['df_master']) if models.get('layer1') else 7374
n_features = len(models.get('feature_cols', [])) or 118

st.markdown(f"""
<div class="header-box">
    <p class="header-title">🧬 RareDx AI <span style="font-size:1rem;color:#845EF7;">v2.0</span></p>
    <p class="header-sub">
        4-Layer Rare Disease Diagnosis Intelligence System
        &nbsp;·&nbsp; MBA Term 8 Project
    </p>
    <span class="header-badge">🔵 L1: Cosine Similarity {l1}</span>
    <span class="header-badge">🟢 L2: Clustering {l2}</span>
    <span class="header-badge">🟠 L3: Knowledge Graph {l3}</span>
    <span class="header-badge">🔴 L4: Anomaly Detection {l4}</span>
    <span class="header-badge">📊 {n_diseases:,} Diseases</span>
    <span class="header-badge">⚙️ {n_features} Features</span>
</div>
""", unsafe_allow_html=True)

# Error if base model missing
if not models.get('layer1'):
    st.error("Base model missing. Run `step3_model_training.py` first.")
    st.stop()

# ============================================
# LAYOUT
# ============================================
col_left, col_right = st.columns([1, 1.7], gap="large")

# ============================================
# LEFT PANEL
# ============================================
with col_left:
    st.markdown('<p class="section-label">Patient Clinical Profile</p>',
                unsafe_allow_html=True)

    # Age of onset
    st.markdown("**Age of Symptom Onset**")
    age_of_onset = st.selectbox(
        "age", [
            "Neonatal", "Antenatal", "Infancy",
            "Childhood", "Adolescent",
            "Adult", "Elderly", "All ages"
        ], index=2, label_visibility="collapsed"
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Inheritance
    st.markdown("**Inheritance Pattern**")
    st.caption("Based on family history")
    inheritance = st.selectbox(
        "inh", [
            "Autosomal recessive",
            "Autosomal dominant",
            "X-linked recessive",
            "X-linked dominant",
            "Mitochondrial inheritance",
            "Multigenic/multifactorial",
            "Not applicable",
            "Unknown"
        ], index=0, label_visibility="collapsed"
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Symptoms (HPO)
    st.markdown("**Patient Symptoms**")
    st.caption("Select all presenting symptoms")
    selected_symptoms = []
    for cat_name, terms in HPO_CATEGORIES.items():
        with st.expander(f"{cat_name}", expanded=False):
            for hpo_id, label in terms.items():
                if st.checkbox(label, key=f"sym_{hpo_id}"):
                    selected_symptoms.append(hpo_id)

    if selected_symptoms:
        tags = "".join([
            f'<span class="hpo-tag">{HPO_FLAT.get(s,s)}</span>'
            for s in selected_symptoms
        ])
        st.markdown(f"**Selected:** {tags}", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Genes
    st.markdown("**Known Gene(s) — Optional**")
    TOP_GENES = [
        "None","HBB","TP53","KIT","LMNA","COL2A1","FGFR1",
        "PIK3CA","BRAF","PTEN","TTN","SHH","FGFR3","IDH2",
        "PAX6","RYR1","FGFR2","NKX2-5","FLNA","DSP","BRCA2",
        "KRAS","COL1A1","LRP5","PTCH1","GNAS","ACTA1","BRCA1",
        "TERT","FBN1","TRPV4","RET","TP63","NF1","CTNNB1",
        "GDF5","COL7A1","TET2","FLT3","SCN5A","ATM","WT1",
        "GATA1","STAT3","GAA","SMN1","FMR1","CFTR","DMD",
        "HTT","MECP2","TSC1","TSC2"
    ]
    cg1, cg2 = st.columns(2)
    with cg1:
        gene1 = st.selectbox("G1", TOP_GENES, 0,
                              label_visibility="collapsed")
    with cg2:
        gene2 = st.selectbox("G2", TOP_GENES, 0,
                              label_visibility="collapsed")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Options
    st.markdown("**Options**")
    top_n = st.slider("Results", 3, 10, 5,
                      label_visibility="collapsed")
    use_clustering = st.toggle(
        "Layer 2: Use Clustering",
        value=models.get('layer2', False),
        disabled=not models.get('layer2', False)
    )
    show_graph = st.toggle(
        "Layer 3: Show Gene Graph Links",
        value=models.get('layer3', False),
        disabled=not models.get('layer3', False)
    )
    check_anomaly = st.toggle(
        "Layer 4: Check Ultra-Rare Flag",
        value=models.get('layer4', False),
        disabled=not models.get('layer4', False)
    )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    predict_clicked = st.button(
        "🔍  Analyse Patient Profile",
        use_container_width=True
    )

    # Stats
    n_sym = len(selected_symptoms)
    st.markdown(f"""
    <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
        <div class="metric-box" style="flex:1;">
            <div class="metric-val">{n_diseases:,}</div>
            <div class="metric-lbl">Diseases</div>
        </div>
        <div class="metric-box" style="flex:1;">
            <div class="metric-val">{n_features}</div>
            <div class="metric-lbl">Features</div>
        </div>
        <div class="metric-box" style="flex:1;">
            <div class="metric-val">{n_sym}</div>
            <div class="metric-lbl">Symptoms</div>
        </div>
        <div class="metric-box" style="flex:1;">
            <div class="metric-val">4</div>
            <div class="metric-lbl">AI Layers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# RIGHT PANEL
# ============================================
with col_right:
    st.markdown('<p class="section-label">Diagnosis Results</p>',
                unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:3rem;margin-bottom:12px;">🧬</div>
            <div style="font-size:1rem;font-weight:600;color:#666688;margin-bottom:8px;">
                Enter patient details and click<br>
                "Analyse Patient Profile"
            </div>
            <div style="font-size:0.82rem;color:#444466;line-height:1.7;">
                <span class="layer-badge layer-1">L1</span>Cosine Similarity retrieval<br>
                <span class="layer-badge layer-2">L2</span>Disease cluster routing<br>
                <span class="layer-badge layer-3">L3</span>Gene knowledge graph links<br>
                <span class="layer-badge layer-4">L4</span>Ultra-rare anomaly detection
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        feature_cols = models['feature_cols']
        X_norm       = models['X_norm']
        df_master    = models['df_master']

        # ── Build patient vector ──────────────────
        onset_score_map = {
            'Antenatal':9,'Neonatal':8,'Infancy':7,
            'Childhood':6,'Adolescent':5,'All ages':4,
            'Adult':3,'Elderly':1
        }
        patient = pd.DataFrame(0, index=[0], columns=feature_cols)
        patient['onset_score']   = onset_score_map.get(age_of_onset, 4)
        patient['rarity_weight'] = 5

        is_neonatal = age_of_onset in ['Neonatal','Antenatal']
        is_ar       = 'Autosomal recessive' in inheritance
        if is_neonatal and is_ar:   patient['urgency_tier'] = 1
        elif age_of_onset in ['Infancy','Childhood']: patient['urgency_tier'] = 2
        else:                       patient['urgency_tier'] = 3

        inh_col = 'inh_' + inheritance.lower()\
                            .replace(' ','_').replace('/','_')
        if inh_col in patient.columns:
            patient[inh_col] = 1

        for s in selected_symptoms:
            hpo_f = 'hpo_' + s.replace(':','_')
            if hpo_f in patient.columns:
                patient[hpo_f] = 1.0

        gene_hints = [g for g in [gene1, gene2] if g != "None"]
        for gene in gene_hints:
            if gene in patient.columns:
                patient[gene] = 1.0

        patient_np   = np.nan_to_num(
            patient.values.astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        patient_norm = normalize(patient_np, norm='l2')

        # ── LAYER 4: Anomaly check ────────────────
        anomaly_result = None
        if check_anomaly and models.get('layer4'):
            score     = float(models['anomaly_model'].score_samples(patient_norm)[0])
            thresh    = models['anomaly_threshold']
            is_anom   = score < thresh

            if score < thresh - 0.1:
                risk = 'EXTREME'
            elif is_anom:
                risk = 'ULTRA-RARE'
            else:
                risk = 'NORMAL'

            anomaly_result = {
                'is_anomaly': is_anom,
                'score'     : score,
                'threshold' : thresh,
                'risk'      : risk
            }

            if is_anom:
                st.markdown(f"""
                <div class="anomaly-banner">
                    <div style="font-size:1rem;font-weight:700;color:#FF4444;margin-bottom:4px;">
                        ⚠️ LAYER 4 ALERT: {risk} DISEASE PATTERN DETECTED
                    </div>
                    <div style="font-size:0.83rem;color:#FFAAAA;">
                        Anomaly score: <strong>{score:.4f}</strong>
                        (threshold: {thresh:.4f})<br>
                        This patient profile matches an ultra-rare disease
                        pattern (top 5% rarest). Immediate referral to a
                        specialist genetic centre is recommended in addition
                        to the results below.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── LAYER 2: Get cluster ──────────────────
        cluster_label = None
        search_X      = X_norm
        search_df     = df_master

        if use_clustering and models.get('layer2'):
            cluster_id = int(models['cluster_model'].predict(patient_norm)[0])
            c_info     = models['cluster_info'].get(cluster_id, {})
            cluster_label = c_info.get('label', f'Cluster {cluster_id}')
            c_size        = c_info.get('size', '?')

            st.markdown(f"""
            <div class="cluster-badge">
                <span class="layer-badge layer-2">L2</span>
                Cluster assigned: <strong>{cluster_label}</strong>
                &nbsp;·&nbsp; {c_size} diseases in this group
                &nbsp;·&nbsp; Searching within cluster for best matches
            </div>
            """, unsafe_allow_html=True)

            # Filter to cluster
            ca = models.get('cluster_assignments')
            if ca is not None:
                cluster_mask = (ca['cluster'] == cluster_id).values
                if cluster_mask.sum() > 5:
                    search_X  = X_norm[cluster_mask]
                    search_df = df_master.iloc[
                        np.where(cluster_mask)[0]
                    ].reset_index(drop=True)

        # ── LAYER 1: Cosine similarity ────────────
        sims        = cosine_similarity(patient_norm, search_X)[0]
        top_indices = sims.argsort()[-top_n:][::-1]

        # ── Patient summary bar ───────────────────
        tc = {1:"#FF4444",2:"#FFA500",3:"#00C9A7"}
        tl = {1:"CRITICAL",2:"URGENT",3:"MONITOR"}
        tv = int(patient['urgency_tier'].values[0])
        genes_str = ", ".join(gene_hints) if gene_hints else "None"
        syms_str  = (", ".join([HPO_FLAT.get(s,s) for s in selected_symptoms[:3]])
                     + ("..." if len(selected_symptoms)>3 else "")
                     ) if selected_symptoms else "None"

        st.markdown(f"""
        <div style="background:#12121F;border:1px solid #2A2A4A;
                    border-radius:10px;padding:12px 16px;margin-bottom:14px;">
            <div style="font-size:0.68rem;color:#555580;letter-spacing:1.5px;
                        text-transform:uppercase;margin-bottom:7px;">
                Patient Profile Summary
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:5px;align-items:center;">
                <span class="info-badge">📅 {age_of_onset}</span>
                <span class="info-badge">🧬 {inheritance}</span>
                <span class="info-badge">🔬 {genes_str}</span>
                <span class="info-badge">🩺 {len(selected_symptoms)} symptoms</span>
                <span style="display:inline-block;background:{tc[tv]}22;
                             border:1px solid {tc[tv]};border-radius:20px;
                             padding:2px 10px;font-size:0.75rem;
                             font-weight:700;color:{tc[tv]};">
                    {tl[tv]}
                </span>
            </div>
            {f'<div style="margin-top:7px;font-size:0.73rem;color:#666688;">{syms_str}</div>' if selected_symptoms else ''}
        </div>
        """, unsafe_allow_html=True)

        # ── Result cards ──────────────────────────
        tier_css    = {1:"result-critical",2:"result-urgent",3:"result-monitor"}
        tier_emoji  = {1:"🔴",2:"🟡",3:"🟢"}
        tier_action = {
            1: "CRITICAL — Refer to metabolic specialist immediately",
            2: "URGENT — Schedule genetic testing within 2 weeks",
            3: "MONITOR — Follow-up with paediatric geneticist"
        }

        for rank, idx in enumerate(top_indices, 1):
            row     = search_df.iloc[idx]
            tier    = int(row['urgency_tier'])
            sim     = float(sims[idx])
            css     = tier_css.get(tier,"result-monitor")
            name    = str(row['DiseaseName'])
            orpha   = int(row['OrphaCode'])
            onset_v = str(row.get('AgeOfOnset','N/A'))
            inh_v   = str(row.get('TypeOfInheritance','N/A'))
            disp    = (name[:62]+"…" if len(name)>62 else name)
            inh_d   = (inh_v[:38]+"…" if len(inh_v)>38 else inh_v)

            # Layer badges on card
            layer_badges = '<span class="layer-badge layer-1">L1</span>'
            if use_clustering and cluster_label:
                layer_badges += '<span class="layer-badge layer-2">L2</span>'

            st.markdown(f"""
            <div class="result-card {css}">
                <div style="display:flex;justify-content:space-between;
                            align-items:flex-start;gap:12px;">
                    <div style="flex:1;min-width:0;">
                        <div style="font-size:0.68rem;color:#666688;
                                    letter-spacing:1px;text-transform:uppercase;
                                    margin-bottom:3px;">
                            #{rank} &nbsp; {layer_badges}
                        </div>
                        <div style="font-size:0.98rem;font-weight:700;
                                    color:#F0F0FF;margin-bottom:4px;"
                             title="{name}">{disp}</div>
                        <div style="font-size:0.76rem;color:#8888AA;">
                            OrphaCode: <strong>{orpha}</strong>
                            &nbsp;·&nbsp; {onset_v[:22]}
                            &nbsp;·&nbsp; {inh_d}
                        </div>
                    </div>
                    <div style="text-align:right;flex-shrink:0;">
                        <div style="font-size:1.5rem;font-weight:800;color:#845EF7;">
                            {sim:.0%}
                        </div>
                        <div style="font-size:0.66rem;color:#555580;">similarity</div>
                    </div>
                </div>
                <div style="margin-top:8px;font-size:0.77rem;
                            padding:5px 9px;border-radius:5px;
                            background:rgba(255,255,255,0.04);color:#AAAACC;">
                    {tier_emoji.get(tier,'')} {tier_action.get(tier,'')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── LAYER 3: Graph neighbours ─────────
            if show_graph and models.get('layer3'):
                graph     = models['disease_graph']
                dis_info  = models.get('disease_info', {})
                nb_raw    = graph.get(int(orpha), {})
                if nb_raw:
                    top3_nb = sorted(
                        nb_raw.items(),
                        key=lambda x: len(x[1]),
                        reverse=True
                    )[:3]
                    nb_html = ""
                    for nb_code, shared in top3_nb:
                        nb_info = dis_info.get(int(nb_code), {})
                        nb_name = str(nb_info.get(
                            'DiseaseName', f'OrphaCode:{nb_code}'
                        ))[:40]
                        genes_shared = ", ".join(shared[:3])
                        nb_html += (
                            f'<span style="display:block;font-size:0.76rem;'
                            f'color:#FFCC88;margin-bottom:2px;">'
                            f'→ {nb_name} '
                            f'<span style="color:#888;font-size:0.7rem;">'
                            f'(shared: {genes_shared})</span>'
                            f'</span>'
                        )
                    st.markdown(f"""
                    <div class="graph-card">
                        <span class="layer-badge layer-3">L3</span>
                        <span style="font-size:0.75rem;color:#FFA500;
                                     font-weight:600;">
                            Related via shared genes:
                        </span>
                        <div style="margin-top:5px;">{nb_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Anomaly normal state note
        if check_anomaly and anomaly_result and not anomaly_result['is_anomaly']:
            st.markdown(f"""
            <div style="background:rgba(0,201,167,0.06);
                        border:1px solid rgba(0,201,167,0.25);
                        border-radius:8px;padding:8px 12px;
                        font-size:0.78rem;color:#00C9A7;margin-top:8px;">
                <span class="layer-badge layer-4">L4</span>
                Anomaly score: {anomaly_result['score']:.4f}
                (threshold {anomaly_result['threshold']:.4f}) —
                Profile within normal rare disease range.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Clinical Disclaimer:</strong>
            This tool is for research and decision support only.
            All diagnoses must be confirmed by a qualified medical geneticist.
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1: st.caption("🧬 RareDx AI v2.0 · MBA Term 8 Project")
with c2: st.caption("📊 Data: Orphanet · HPO · orphadata.com")
with c3: st.caption("⚠️ For research use only · Not clinical diagnosis")
