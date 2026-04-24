# ============================================
# RARE DISEASE AI PROJECT
# Step 4: Doctor Interface (COMPLETE + HPO)
# ============================================
# Run with:
#   streamlit run step4_doctor_interface.py
#
# Requires these files in same folder:
#   output_model.pkl
#   output_feature_cols.pkl
#   output_master_features.csv
#   output_hpo_terms.csv (if step3b was run)
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
    page_title = "RareDx AI — Rare Disease Diagnosis Assistant",
    page_icon  = "🧬",
    layout     = "wide"
)

# ============================================
# STYLING
# ============================================
st.markdown("""
<style>
    /* General */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #845EF7;
        border-radius: 14px;
        padding: 22px 30px;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #F0F0FF;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .header-sub {
        color: #8888AA;
        font-size: 0.88rem;
        margin: 0;
    }
    .header-badge {
        display: inline-block;
        background: rgba(132,94,247,0.15);
        border: 1px solid rgba(132,94,247,0.4);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        color: #A78BFA;
        margin-right: 8px;
        margin-top: 10px;
    }

    /* Section headers */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #845EF7;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* Result cards */
    .result-card {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 5px solid;
        transition: all 0.2s;
    }
    .result-critical {
        background: rgba(255,68,68,0.08);
        border-color: #FF4444;
    }
    .result-urgent {
        background: rgba(255,165,0,0.08);
        border-color: #FFA500;
    }
    .result-monitor {
        background: rgba(0,201,167,0.08);
        border-color: #00C9A7;
    }
    .result-rank {
        font-size: 0.7rem;
        color: #666688;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 3px;
    }
    .result-name {
        font-size: 1rem;
        font-weight: 700;
        color: #F0F0FF;
        margin-bottom: 5px;
    }
    .result-meta {
        font-size: 0.78rem;
        color: #8888AA;
    }
    .result-score {
        font-size: 1.6rem;
        font-weight: 800;
        color: #845EF7;
    }
    .result-score-label {
        font-size: 0.68rem;
        color: #555580;
    }
    .result-action {
        margin-top: 8px;
        font-size: 0.78rem;
        padding: 5px 10px;
        border-radius: 6px;
        background: rgba(255,255,255,0.04);
        color: #AAAACC;
    }

    /* Metric boxes */
    .metric-row {
        display: flex;
        gap: 10px;
        margin-top: 14px;
    }
    .metric-box {
        flex: 1;
        background: #1A1A2E;
        border: 1px solid #2A2A4A;
        border-radius: 8px;
        padding: 10px 6px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #845EF7;
    }
    .metric-lbl {
        font-size: 0.65rem;
        color: #555580;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Info badge */
    .info-badge {
        display: inline-block;
        background: #1A1A2E;
        border: 1px solid #333366;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.78rem;
        color: #8888CC;
        margin-right: 5px;
        margin-bottom: 4px;
    }
    .urgency-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 5px;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #845EF7, #5C3D99) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 32px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #9B6FFF, #7050BB) !important;
        transform: translateY(-1px);
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255,165,0,0.07);
        border: 1px solid rgba(255,165,0,0.3);
        border-radius: 8px;
        padding: 10px 14px;
        color: #CC8800;
        font-size: 0.8rem;
        margin-top: 14px;
    }

    /* Empty state */
    .empty-state {
        background: #12121F;
        border: 1px dashed #2A2A4A;
        border-radius: 12px;
        padding: 56px 32px;
        text-align: center;
    }

    /* HPO tag */
    .hpo-tag {
        display: inline-block;
        background: rgba(0,201,167,0.1);
        border: 1px solid rgba(0,201,167,0.3);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        color: #00C9A7;
        margin: 2px;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #1E1E3A;
        margin: 14px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HPO SYMPTOM DICTIONARY
# ============================================
# Comprehensive readable symptom list
# Used as fallback if output_hpo_terms.csv
# is not available (before step3b is run)
# ============================================
HPO_SYMPTOM_DICT = {
    # Neurological
    "HP:0001250": "Seizures",
    "HP:0001251": "Ataxia (loss of coordination)",
    "HP:0001257": "Spasticity",
    "HP:0001263": "Global developmental delay",
    "HP:0001290": "Hypotonia (low muscle tone)",
    "HP:0001252": "Muscular hypotonia",
    "HP:0001324": "Muscle weakness",
    "HP:0002011": "Morphological abnormality of the CNS",
    "HP:0001256": "Intellectual disability",
    "HP:0000752": "Hyperactivity",
    "HP:0002094": "Dyspnea (shortness of breath)",
    "HP:0001272": "Cerebellar atrophy",
    "HP:0001305": "Dandy-Walker malformation",
    "HP:0002059": "Cerebral atrophy",
    "HP:0001317": "Abnormal cerebellum morphology",

    # Muscular
    "HP:0003326": "Myalgia (muscle pain)",
    "HP:0003560": "Muscular dystrophy",
    "HP:0003457": "EMG abnormality",
    "HP:0002529": "Neuronal loss",
    "HP:0003236": "Elevated CK levels",

    # Visual
    "HP:0000572": "Visual loss",
    "HP:0000639": "Nystagmus",
    "HP:0000478": "Abnormality of the eye",
    "HP:0000486": "Strabismus",
    "HP:0001087": "Retinal degeneration",
    "HP:0007663": "Reduced visual acuity",
    "HP:0000598": "Abnormal ear morphology",
    "HP:0000505": "Visual impairment",

    # Hearing
    "HP:0000365": "Hearing impairment",
    "HP:0000407": "Sensorineural hearing loss",
    "HP:0001319": "Neonatal hypotonia",

    # Skeletal
    "HP:0002650": "Scoliosis",
    "HP:0001166": "Arachnodactyly (long fingers)",
    "HP:0002808": "Kyphosis",
    "HP:0002812": "Coxa vara",
    "HP:0000926": "Platyspondyly",
    "HP:0001373": "Joint dislocation",
    "HP:0002857": "Genu valgum (knock knees)",
    "HP:0003502": "Mild short stature",
    "HP:0004322": "Short stature",
    "HP:0000268": "Dolichocephaly",

    # Cardiac
    "HP:0001638": "Cardiomyopathy",
    "HP:0001626": "Abnormality of the cardiovascular system",
    "HP:0001644": "Dilated cardiomyopathy",
    "HP:0001631": "Atrial septal defect",
    "HP:0001649": "Tachycardia",
    "HP:0001250": "Arrhythmia",

    # Metabolic
    "HP:0001508": "Failure to thrive",
    "HP:0001943": "Hypoglycemia",
    "HP:0002017": "Nausea and vomiting",
    "HP:0000822": "Hypertension",
    "HP:0003128": "Lactic acidosis",
    "HP:0001985": "Hypoketotic hypoglycemia",

    # Skin
    "HP:0000964": "Skin abnormality",
    "HP:0001000": "Abnormality of skin pigmentation",
    "HP:0007565": "Multiple cafe-au-lait spots",
    "HP:0001030": "Fragile skin",
    "HP:0000988": "Skin rash",

    # Facial / Dysmorphic
    "HP:0000272": "Malar flattening",
    "HP:0000316": "Hypertelorism (wide-set eyes)",
    "HP:0000175": "Cleft palate",
    "HP:0000276": "Long face",
    "HP:0000303": "Mandibular prognathia",
    "HP:0000343": "Long philtrum",
    "HP:0000369": "Low-set ears",
    "HP:0000431": "Wide nasal bridge",

    # Respiratory
    "HP:0002088": "Abnormal lung morphology",
    "HP:0000961": "Cyanosis",
    "HP:0002093": "Respiratory insufficiency",
    "HP:0006536": "Obstructive lung disease",

    # Renal
    "HP:0000077": "Abnormality of the kidney",
    "HP:0000107": "Renal cyst",
    "HP:0000093": "Proteinuria",
    "HP:0003774": "Stage 5 chronic kidney disease",
}

# Group symptoms by category for UI
HPO_CATEGORIES = {
    "🧠 Neurological": [
        "HP:0001250", "HP:0001251", "HP:0001257", "HP:0001263",
        "HP:0001290", "HP:0001252", "HP:0001324", "HP:0001256",
        "HP:0000752", "HP:0001272", "HP:0002059"
    ],
    "💪 Muscular": [
        "HP:0003326", "HP:0003560", "HP:0003236", "HP:0003457"
    ],
    "👁️ Visual": [
        "HP:0000572", "HP:0000639", "HP:0000478", "HP:0000486",
        "HP:0001087", "HP:0000505", "HP:0007663"
    ],
    "👂 Hearing": [
        "HP:0000365", "HP:0000407"
    ],
    "🦴 Skeletal": [
        "HP:0002650", "HP:0001166", "HP:0002808", "HP:0004322",
        "HP:0003502", "HP:0001373", "HP:0002857"
    ],
    "❤️ Cardiac": [
        "HP:0001638", "HP:0001644", "HP:0001631", "HP:0001649"
    ],
    "⚗️ Metabolic": [
        "HP:0001508", "HP:0001943", "HP:0003128", "HP:0001985"
    ],
    "🫁 Respiratory": [
        "HP:0002088", "HP:0002093", "HP:0000961"
    ],
    "🧬 Skin": [
        "HP:0000964", "HP:0007565", "HP:0001030", "HP:0000988"
    ],
    "👤 Facial / Dysmorphic": [
        "HP:0000316", "HP:0000175", "HP:0000369", "HP:0000431"
    ],
    "🫘 Renal": [
        "HP:0000077", "HP:0000107", "HP:0000093"
    ],
}

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    try:
        with open('output_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        with open('output_feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        df_master = pd.read_csv('output_master_features.csv')

        X_raw = np.nan_to_num(
            df_master[feature_cols].values.astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        X_norm = normalize(X_raw, norm='l2')

        # Check if HPO features exist
        hpo_cols = [c for c in feature_cols if c.startswith('hpo_')]

        # Load HPO terms if available
        try:
            hpo_terms_df = pd.read_csv('output_hpo_terms.csv')
            hpo_available = True
        except:
            hpo_terms_df = None
            hpo_available = len(hpo_cols) > 0

        return (model_data, feature_cols, df_master,
                X_norm, hpo_cols, hpo_available, None)

    except FileNotFoundError as e:
        return None, None, None, None, [], False, str(e)

(model_data, feature_cols, df_master,
 X_norm, hpo_cols, hpo_available, load_error) = load_model()

# ============================================
# HEADER
# ============================================
hpo_status = "HPO Symptoms ✅" if hpo_available else "HPO Symptoms ⚠️ Run step3b first"
n_diseases = len(df_master) if df_master is not None else 7374
n_features = len(feature_cols) if feature_cols else 118

st.markdown(f"""
<div class="header-box">
    <p class="header-title">🧬 RareDx AI</p>
    <p class="header-sub">
        AI-Powered Rare Disease Diagnosis Assistant
        &nbsp;·&nbsp; MBA Term 8 Project
        &nbsp;·&nbsp; Powered by Cosine Similarity Retrieval
    </p>
    <span class="header-badge">📊 {n_diseases:,} Diseases</span>
    <span class="header-badge">⚙️ {n_features:,} Features</span>
    <span class="header-badge">🧠 {hpo_status}</span>
    <span class="header-badge">📍 Orphanet Database</span>
</div>
""", unsafe_allow_html=True)

# ============================================
# ERROR STATE
# ============================================
if load_error:
    st.error(f"""
**⚠️ Model files not found. Please run Step 3 first:**
```
python step3_model_training.py
```
Then relaunch this app.

Error: `{load_error}`
    """)
    st.stop()

# ============================================
# LAYOUT
# ============================================
col_left, col_right = st.columns([1, 1.7], gap="large")

# ============================================
# LEFT PANEL — INPUT FORM
# ============================================
with col_left:
    st.markdown('<p class="section-label">Patient Clinical Profile</p>',
                unsafe_allow_html=True)

    # ── Age of Onset ──────────────────────────
    st.markdown("**Age of Symptom Onset**")
    age_of_onset = st.selectbox(
        label            = "age",
        options          = [
            "Neonatal", "Antenatal", "Infancy",
            "Childhood", "Adolescent",
            "Adult", "Elderly", "All ages"
        ],
        index            = 2,
        label_visibility = "collapsed"
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Inheritance Pattern ────────────────────
    st.markdown("**Inheritance Pattern**")
    st.caption("Based on family history")
    inheritance = st.selectbox(
        label            = "inh",
        options          = [
            "Autosomal recessive",
            "Autosomal dominant",
            "X-linked recessive",
            "X-linked dominant",
            "Mitochondrial inheritance",
            "Multigenic/multifactorial",
            "Not applicable",
            "Unknown"
        ],
        index            = 0,
        label_visibility = "collapsed"
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Symptoms (HPO) ─────────────────────────
    st.markdown("**Patient Symptoms**")
    st.caption(
        "Select all symptoms the patient presents. "
        + ("HPO features active ✅" if hpo_available
           else "⚠️ Run step3b_hpo_integration.py for full accuracy")
    )

    selected_symptoms = []

    # Group by category using expanders
    for category, hpo_ids in HPO_CATEGORIES.items():
        with st.expander(category, expanded=False):
            for hpo_id in hpo_ids:
                label = HPO_SYMPTOM_DICT.get(hpo_id, hpo_id)
                if st.checkbox(label, key=f"sym_{hpo_id}"):
                    selected_symptoms.append(hpo_id)

    if selected_symptoms:
        st.markdown(
            "**Selected:** " +
            "".join([
                f'<span class="hpo-tag">'
                f'{HPO_SYMPTOM_DICT.get(s, s)}</span>'
                for s in selected_symptoms
            ]),
            unsafe_allow_html=True
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Gene Hints ─────────────────────────────
    st.markdown("**Known Gene(s) — Optional**")
    st.caption("If genetic test result is available")

    TOP_GENES = [
        "None", "HBB", "TP53", "KIT", "LMNA", "COL2A1",
        "FGFR1", "PIK3CA", "BRAF", "PTEN", "TTN", "SHH",
        "FGFR3", "IDH2", "PAX6", "RYR1", "FGFR2", "NKX2-5",
        "FLNA", "DSP", "BRCA2", "KRAS", "COL1A1", "LRP5",
        "PTCH1", "GNAS", "ACTA1", "BRCA1", "TERT", "FBN1",
        "TRPV4", "RET", "TP63", "NF1", "CTNNB1", "GDF5",
        "COL7A1", "TET2", "ABCB6", "FLT3", "SCN5A", "ATM",
        "HLA-DRB1", "HBA2", "TBC1D24", "WT1", "GATA1",
        "GNA11", "GATA4", "ARX", "STAT3", "GAA", "SMN1",
        "FMR1", "CFTR", "DMD", "HTT", "MECP2", "TSC1", "TSC2"
    ]

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        gene1 = st.selectbox("Gene 1", TOP_GENES,
                             index=0, label_visibility="collapsed")
    with col_g2:
        gene2 = st.selectbox("Gene 2", TOP_GENES,
                             index=0, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Number of Results ──────────────────────
    st.markdown("**Number of Results to Show**")
    top_n = st.slider(
        label            = "topn",
        min_value        = 3,
        max_value        = 10,
        value            = 5,
        label_visibility = "collapsed"
    )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Predict Button ─────────────────────────
    predict_clicked = st.button(
        "🔍  Find Matching Diseases",
        use_container_width=True
    )

    # ── Stats ──────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-val">{n_diseases:,}</div>
            <div class="metric-lbl">Diseases</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{n_features}</div>
            <div class="metric-lbl">Features</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{len(selected_symptoms)}</div>
            <div class="metric-lbl">Symptoms</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">
                {"✅" if hpo_available else "⚠️"}
            </div>
            <div class="metric-lbl">HPO</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# RIGHT PANEL — RESULTS
# ============================================
with col_right:
    st.markdown('<p class="section-label">Diagnosis Results</p>',
                unsafe_allow_html=True)

    # ── Empty State ────────────────────────────
    if not predict_clicked:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:3.5rem; margin-bottom:14px;">🧬</div>
            <div style="font-size:1.05rem; font-weight:600;
                        color:#666688; margin-bottom:8px;">
                Enter patient details and click<br>
                "Find Matching Diseases"
            </div>
            <div style="font-size:0.83rem; color:#444466;
                        line-height:1.6;">
                The AI searches <strong style="color:#845EF7">7,374
                rare disease profiles</strong><br>
                using symptoms, genes, onset, and inheritance pattern<br>
                to find the closest clinical matches instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Results ────────────────────────────────
    else:
        # Build patient feature vector
        onset_score_map = {
            'Antenatal':9, 'Neonatal':8, 'Infancy':7,
            'Childhood':6, 'Adolescent':5, 'All ages':4,
            'Adult':3,     'Elderly':1
        }

        patient = pd.DataFrame(0, index=[0], columns=feature_cols)

        # Onset score
        patient['onset_score']   = onset_score_map.get(age_of_onset, 4)
        patient['rarity_weight'] = 5

        # Urgency tier
        is_neonatal = age_of_onset in ['Neonatal', 'Antenatal']
        is_ar       = 'Autosomal recessive' in inheritance
        if is_neonatal and is_ar:
            patient['urgency_tier'] = 1
        elif age_of_onset in ['Infancy', 'Childhood']:
            patient['urgency_tier'] = 2
        else:
            patient['urgency_tier'] = 3

        # Inheritance
        inh_col = ('inh_' + inheritance.lower()
                   .replace(' ', '_').replace('/', '_'))
        if inh_col in patient.columns:
            patient[inh_col] = 1

        # HPO symptoms
        for sym_id in selected_symptoms:
            hpo_feat = 'hpo_' + sym_id.replace(':', '_')
            if hpo_feat in patient.columns:
                patient[hpo_feat] = 1.0

        # Gene hints
        gene_hints = [g for g in [gene1, gene2] if g != "None"]
        for gene in gene_hints:
            if gene in patient.columns:
                patient[gene] = 1.0

        # Normalize and compute similarity
        patient_np   = np.nan_to_num(
            patient.values.astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        patient_norm = normalize(patient_np, norm='l2')
        sims         = cosine_similarity(patient_norm, X_norm)[0]
        top_indices  = sims.argsort()[-top_n:][::-1]

        # Patient summary bar
        tier_val     = int(patient['urgency_tier'].values[0])
        tier_color   = {1: "#FF4444", 2: "#FFA500", 3: "#00C9A7"}
        tier_label   = {1: "🔴 CRITICAL", 2: "🟡 URGENT", 3: "🟢 MONITOR"}
        genes_str    = ", ".join(gene_hints) if gene_hints else "None"
        symptoms_str = (", ".join([HPO_SYMPTOM_DICT.get(s, s)
                                   for s in selected_symptoms[:3]])
                        + ("..." if len(selected_symptoms) > 3 else "")
                        ) if selected_symptoms else "None selected"

        tc = tier_color[tier_val]
        st.markdown(f"""
        <div style="background:#12121F; border:1px solid #2A2A4A;
                    border-radius:10px; padding:12px 16px;
                    margin-bottom:16px;">
            <div style="font-size:0.7rem; color:#555580;
                        letter-spacing:1.5px; text-transform:uppercase;
                        margin-bottom:8px;">
                Patient Profile Summary
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:6px;
                        align-items:center;">
                <span class="info-badge">📅 {age_of_onset}</span>
                <span class="info-badge">🧬 {inheritance}</span>
                <span class="info-badge">🔬 {genes_str}</span>
                <span class="info-badge">🩺 {len(selected_symptoms)} symptoms</span>
                <span class="urgency-badge" style="
                    background:{tc}22;
                    border:1px solid {tc};
                    color:{tc};">
                    {tier_label[tier_val]}
                </span>
            </div>
            {f'<div style="margin-top:8px; font-size:0.75rem; color:#666688;">Symptoms: {symptoms_str}</div>' if selected_symptoms else ''}
        </div>
        """, unsafe_allow_html=True)

        # Result cards
        tier_css    = {1: "result-critical",
                       2: "result-urgent",
                       3: "result-monitor"}
        tier_emoji  = {1: "🔴", 2: "🟡", 3: "🟢"}
        tier_action = {
            1: "CRITICAL — Refer to metabolic/genetic specialist immediately",
            2: "URGENT — Schedule genetic testing within 2 weeks",
            3: "MONITOR — Follow-up with paediatric geneticist"
        }

        for rank, idx in enumerate(top_indices, 1):
            row          = df_master.iloc[idx]
            tier         = int(row['urgency_tier'])
            sim_score    = float(sims[idx])
            css_class    = tier_css.get(tier, "result-monitor")
            disease_name = str(row['DiseaseName'])
            orpha_code   = int(row['OrphaCode'])
            onset_val    = str(row.get('AgeOfOnset', 'N/A'))
            inh_val      = str(row.get('TypeOfInheritance', 'N/A'))
            action       = tier_action.get(tier, "")

            # Truncate long names
            display_name = (disease_name[:65] + "…"
                            if len(disease_name) > 65
                            else disease_name)
            # Truncate long inheritance
            inh_display  = (inh_val[:40] + "…"
                            if len(inh_val) > 40 else inh_val)

            st.markdown(f"""
            <div class="result-card {css_class}">
                <div style="display:flex; justify-content:space-between;
                            align-items:flex-start; gap:12px;">
                    <div style="flex:1; min-width:0;">
                        <div class="result-rank">#{rank} Match</div>
                        <div class="result-name"
                             title="{disease_name}">
                            {display_name}
                        </div>
                        <div class="result-meta">
                            OrphaCode: <strong>{orpha_code}</strong>
                            &nbsp;·&nbsp; Onset: {onset_val}
                            &nbsp;·&nbsp; {inh_display}
                        </div>
                    </div>
                    <div style="text-align:right; flex-shrink:0;">
                        <div class="result-score">
                            {sim_score:.0%}
                        </div>
                        <div class="result-score-label">
                            similarity
                        </div>
                    </div>
                </div>
                <div class="result-action">
                    {tier_emoji.get(tier,'')} {action}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # HPO status note
        if not hpo_available and selected_symptoms:
            st.markdown("""
            <div style="background:rgba(132,94,247,0.08);
                        border:1px solid rgba(132,94,247,0.3);
                        border-radius:8px; padding:10px 14px;
                        font-size:0.8rem; color:#A78BFA;
                        margin-top:10px;">
                💡 <strong>Improve accuracy:</strong>
                Run <code>step3b_hpo_integration.py</code> to activate
                HPO symptom matching. Your symptom selections will then
                directly influence the similarity scores.
            </div>
            """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Clinical Disclaimer:</strong>
            This tool is for research and decision support only.
            All diagnoses must be confirmed by a qualified medical
            geneticist. Not a substitute for clinical judgment.
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("🧬 RareDx AI · MBA Term 8 Project")
with col_f2:
    st.caption("📊 Data: Orphanet · HPO · orphadata.com")
with col_f3:
    st.caption("⚠️ For research use only · Not for clinical diagnosis")
