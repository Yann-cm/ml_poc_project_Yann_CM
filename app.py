import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ToxScreen — Aide au diagnostic",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background: #f7f8fc; }
.header-bar { background:#1a2744; border-radius:12px; padding:18px 28px; margin-bottom:24px; display:flex; align-items:center; gap:16px; }
.header-bar h1 { color:#fff; font-size:22px; margin:0; font-weight:600; }
.header-bar p  { color:#8899bb; font-size:13px; margin:0; }
.card { background:white; border-radius:12px; padding:20px 22px; border:1px solid #e8ecf4; margin-bottom:14px; }
.card-title { font-size:12px; font-weight:600; color:#6b7a99; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:14px; }
.verdict-high   { background:#fff2f2; border:2px solid #ff6b6b; border-radius:12px; padding:20px 24px; }
.verdict-medium { background:#fffbf0; border:2px solid #ffa94d; border-radius:12px; padding:20px 24px; }
.verdict-low    { background:#f0fdf4; border:2px solid #51cf66; border-radius:12px; padding:20px 24px; }
.badge-high   { background:#ff6b6b; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-medium { background:#ffa94d; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-low    { background:#51cf66; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.gauge-wrap { background:#e8ecf4; border-radius:6px; height:8px; margin:6px 0; }
.gauge-fill { height:100%; border-radius:6px; }
.metric-box { background:white; border:1px solid #e8ecf4; border-radius:10px; padding:14px 16px; text-align:center; }
.metric-val { font-size:26px; font-weight:700; color:#1a2744; }
.metric-lbl { font-size:11px; color:#6b7a99; margin-top:2px; }
.disclaimer { background:#f0f4ff; border-left:3px solid #4c6ef5; border-radius:0 8px 8px 0; padding:10px 14px; font-size:11px; color:#5c6b8a; margin-top:16px; }
.patient-found { background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:12px 16px; margin-bottom:12px; }
.patient-new   { background:#eff6ff; border:1px solid #93c5fd; border-radius:10px; padding:12px 16px; margin-bottom:12px; }
section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  JSON DATABASE
# ─────────────────────────────────────────
DB_PATH = "patients_db.json"

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def patient_key(prenom, nom):
    return f"{prenom.strip().lower()}_{nom.strip().lower()}"

def get_patient(prenom, nom):
    db  = load_db()
    key = patient_key(prenom, nom)
    return db.get(key, None)

def save_patient(prenom, nom, profile, all_scores, verdict_sub, verdict_score):
    db  = load_db()
    key = patient_key(prenom, nom)
    # On conserve l'historique des analyses
    entry = db.get(key, {"prenom": prenom, "nom": nom, "analyses": []})
    entry["prenom"]  = prenom
    entry["nom"]     = nom
    entry["profile"] = profile  # derniers paramètres sauvegardés
    entry["analyses"].append({
        "verdict_sub":   verdict_sub,
        "verdict_score": round(verdict_score, 4),
        "scores":        {k: round(v, 4) for k, v in all_scores.items()},
    })
    db[key] = entry
    save_db(db)
    return entry

def list_patients():
    db = load_db()
    return [(v["prenom"], v["nom"]) for v in db.values()]

# ─────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('MODELE_SAVE/multiclass_model.pkl', 'rb') as f:
        mc = pickle.load(f)
    with open('MODELE_SAVE/binary_models.pkl', 'rb') as f:
        bin_ = pickle.load(f)
    return mc, bin_

mc_data, bin_data = load_models()
rf_multi     = mc_data['model']
mc_classes   = mc_data['classes']
FEATURE_COLS = bin_data['feature_cols']
FEAT_LABELS  = bin_data['feature_labels']
SUB_LABELS   = bin_data['substance_labels']
TARGET_IL    = bin_data['target_illicit']
models_bin   = bin_data['models']
metrics_bin  = bin_data['metrics']

# Encodages
AGE_ENC = {'< 18 ans':-0.95197,'18-24 ans':0.49788,'25-34 ans':-0.07854,
            '35-44 ans':-0.05921,'45-54 ans':-0.45174,'55-64 ans':1.82213,'65+ ans':2.59171}
GEN_ENC = {'Homme':0.48246,'Femme':-0.48246}
EDU_ENC = {'Inférieur bac':-2.43591,'Bac':-1.73790,'Bac +2':-1.43103,
            'Bac +3':-0.61113,'Bac +5':-0.05921,'Master':0.45468,'Doctorat':1.16365}
PAY_ENC = {'Australie':-0.09765,'Canada':0.24923,'Irlande':0.21128,
            'Nouvelle-Zélande':-0.46841,'UK':0.96082,'USA':-0.57009,'Autre':-0.28519}
ETH_ENC = {'Asiatique':-0.50212,'Noir':-1.10702,'Blanc':-0.31685,
            'Mixte':1.90725,'Autre':0.11440}

AGE_LIST = list(AGE_ENC.keys())
GEN_LIST = list(GEN_ENC.keys())
EDU_LIST = list(EDU_ENC.keys())
PAY_LIST = list(PAY_ENC.keys())
ETH_LIST = list(ETH_ENC.keys())

SUB_INFO = {
    'cannabis':  {'emoji':'🌿','cat':'Cannabinoïdes'},
    'benzos':    {'emoji':'💊','cat':'Anxiolytiques'},
    'ecstasy':   {'emoji':'🔮','cat':'Empathogens'},
    'amphet':    {'emoji':'⚡','cat':'Stimulants'},
    'coke':      {'emoji':'❄️','cat':'Stimulants'},
    'lsd':       {'emoji':'🌀','cat':'Psychédéliques'},
    'mushrooms': {'emoji':'🍄','cat':'Psychédéliques'},
    'legalh':    {'emoji':'🧪','cat':'NPS'},
    'amyl':      {'emoji':'💨','cat':'Inhalants'},
    'meth':      {'emoji':'🔥','cat':'Stimulants'},
    'ketamine':  {'emoji':'🌊','cat':'Dissociatifs'},
    'heroin':    {'emoji':'💉','cat':'Opioïdes'},
    'vsa':       {'emoji':'🫧','cat':'Solvants'},
}

INTERP = {
    'nscore':    ('Neuroticisme élevé','Usage comme mécanisme de régulation émotionnelle'),
    'escore':    ('Extraversion élevée','Forte exposition sociale, contextes festifs'),
    'oscore':    ('Ouverture élevée','Attrait pour les nouvelles expériences et états altérés'),
    'ascore':    ('Agréabilité faible','Moins de résistance à la pression sociale'),
    'cscore':    ('Conscienciosité faible','Moindre contrôle inhibiteur'),
    'impulsive': ('Impulsivité élevée','Initiation rapide sans évaluation des risques'),
    'ss':        ('Sensation-Seeking élevé','Facteur de risque majeur'),
    'age':       ("Tranche d'âge",'Prévalence variable selon la tranche'),
    'gender':    ('Genre','Différences de prévalence documentées'),
    'education': ("Niveau d'éducation","Corrélé à l'exposition aux substances"),
    'country':   ('Pays','Contexte légal et disponibilité'),
    'ethnicity': ('Ethnicité','Facteur socio-culturel'),
    'alcohol_bin':  ("Consommation d'alcool",'Gateway substance'),
    'caff_bin':     ('Caféine','Profil psychoactif quotidien'),
    'choc_bin':     ('Chocolat','Profil hédonique'),
    'nicotine_bin': ('Tabagisme','Corrélé à plusieurs substances illicites'),
}

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if 'history'        not in st.session_state: st.session_state['history']        = []
if 'loaded_patient' not in st.session_state: st.session_state['loaded_patient'] = None
if 'prenom_input'   not in st.session_state: st.session_state['prenom_input']   = ""
if 'nom_input'      not in st.session_state: st.session_state['nom_input']      = ""

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def compute_predictions(X_in):
    probas_mc  = rf_multi.predict_proba(X_in)[0]
    all_scores = {}
    results    = []
    for sub in TARGET_IL:
        if sub not in models_bin: continue
        pmc       = probas_mc[list(mc_classes).index(sub)] if sub in mc_classes else 0.0
        pbin      = models_bin[sub].predict_proba(X_in)[0][1]
        score     = 0.4 * pmc + 0.6 * pbin
        all_scores[sub] = score
        coefs     = np.array(metrics_bin[sub]['coefficients'])
        top_f     = FEATURE_COLS[np.argmax(np.abs(coefs))]
        results.append({'sub':sub,'pmc':pmc,'pbin':pbin,'score':score,
                        'top_f':top_f,'coef':coefs[np.argmax(np.abs(coefs))]})
    results.sort(key=lambda r: r['score'], reverse=True)
    return results, all_scores

def score_color(v):
    if v >= 0.60: return '#ff6b6b', '#fff2f2'
    elif v >= 0.40: return '#d97706', '#fffbf0'
    return '#059669', '#f0fdf4'

def verdict_style(sc):
    if sc >= 0.60: return 'verdict-high',   '⚠️', 'RISQUE ÉLEVÉ',  'badge-high'
    elif sc >= 0.40: return 'verdict-medium','🔶', 'RISQUE MODÉRÉ', 'badge-medium'
    return 'verdict-low', '✅', 'RISQUE FAIBLE', 'badge-low'

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class='header-bar'>
    <div style='font-size:32px'>🏥</div>
    <div>
        <h1>ToxScreen — Aide au diagnostic de consommation</h1>
        <p>Pipeline ML deux étapes · UCI Drug Consumption · Usage académique et préventif uniquement</p>
    </div>
</div>
""", unsafe_allow_html=True)

n_hist = len(st.session_state['history'])
n_db   = len(load_db())
col_stat1, col_stat2, col_stat3 = st.columns([1,1,8])
col_stat1.caption(f"📋 {n_hist} patient(s) en session")
col_stat2.caption(f"🗂️ {n_db} patient(s) en base")

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋  Formulaire patient",
    "🔬  Analyse détaillée",
    "📊  Explicabilité",
    "📁  Historique & Matrice",
    "🧬  Profils de population",
    "ℹ️  Méthodologie",
])

# ═══════════════════════════════════════════
#  TAB 1 — FORMULAIRE
# ═══════════════════════════════════════════
with tab1:

    # ── Recherche patient
    st.markdown("#### Identification du patient")
    id_c1, id_c2, id_c3 = st.columns([2, 2, 2])
    prenom_in = id_c1.text_input("Prénom", value=st.session_state['prenom_input'], placeholder="Ex : Marc")
    nom_in    = id_c2.text_input("Nom",    value=st.session_state['nom_input'],    placeholder="Ex : Dupont")

    with id_c3:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Rechercher / Charger", use_container_width=True)

    # Auto-complétion depuis la DB
    existing_patients = list_patients()
    if existing_patients:
        existing_labels = [f"{p[0]} {p[1]}" for p in existing_patients]
        st.caption(f"💾 Patients enregistrés : {', '.join(existing_labels)}")

    # Chargement
    loaded = None
    if search_clicked and prenom_in.strip() and nom_in.strip():
        loaded = get_patient(prenom_in.strip(), nom_in.strip())
        st.session_state['loaded_patient'] = loaded
        st.session_state['prenom_input']   = prenom_in.strip()
        st.session_state['nom_input']      = nom_in.strip()

    loaded = st.session_state.get('loaded_patient', None)

    if loaded:
        p = loaded['profile']
        st.markdown(f"""
        <div class='patient-found'>
            ✅ <strong>{loaded['prenom']} {loaded['nom']}</strong> trouvé en base —
            profil chargé automatiquement ·
            {len(loaded.get('analyses', []))} analyse(s) précédente(s) enregistrée(s)
        </div>
        """, unsafe_allow_html=True)
    elif prenom_in.strip() and nom_in.strip() and search_clicked:
        st.markdown(f"""
        <div class='patient-new'>
            🆕 <strong>{prenom_in.strip()} {nom_in.strip()}</strong> — nouveau patient.
            Le profil sera enregistré après analyse.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Formulaire avec pré-remplissage si patient chargé
    p = loaded['profile'] if loaded else None

    def pv(key, default):
        """Récupère la valeur sauvegardée ou retourne le défaut."""
        return p[key] if p and key in p else default

    with st.form("patient_form"):
        st.markdown("<div class='form-section-title'>① PROFIL DÉMOGRAPHIQUE</div>", unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        age       = c1.selectbox('Âge',        AGE_LIST, index=AGE_LIST.index(pv('age','18-24 ans')))
        gender    = c2.selectbox('Genre',       GEN_LIST, index=GEN_LIST.index(pv('gender','Homme')))
        education = c3.selectbox('Éducation',   EDU_LIST, index=EDU_LIST.index(pv('education','Bac +3')))
        country   = c4.selectbox('Pays',        PAY_LIST, index=PAY_LIST.index(pv('country','UK')))
        ethnicity = c5.selectbox('Ethnicité',   ETH_LIST, index=ETH_LIST.index(pv('ethnicity','Blanc')))

        st.markdown("<br><div class='form-section-title'>② TRAITS DE PERSONNALITÉ <span style='font-weight:400;color:#9aa5c0;font-size:11px;'>— Score normalisé : −3 (très faible) → +3 (très élevé)</span></div>", unsafe_allow_html=True)
        p1,p2,p3,p4 = st.columns(4)
        nscore    = p1.slider('Neuroticisme',     -3.0, 3.0, float(pv('nscore',   0.0)), 0.1)
        escore    = p1.slider('Extraversion',     -3.0, 3.0, float(pv('escore',   0.0)), 0.1)
        oscore    = p2.slider('Ouverture',        -3.0, 3.0, float(pv('oscore',   0.0)), 0.1)
        ascore    = p2.slider('Agréabilité',      -3.0, 3.0, float(pv('ascore',   0.0)), 0.1)
        cscore    = p3.slider('Conscienciosité',  -3.0, 3.0, float(pv('cscore',   0.0)), 0.1)
        impulsive = p3.slider('Impulsivité',      -3.0, 3.0, float(pv('impulsive',0.0)), 0.1)
        ss        = p4.slider('Sensation-Seeking',-3.0, 3.0, float(pv('ss',       0.0)), 0.1)

        st.markdown("<br><div class='form-section-title'>③ CONSOMMATIONS LÉGALES DÉCLARÉES</div>", unsafe_allow_html=True)
        q1,q2,q3,q4 = st.columns(4)
        alc_bin   = int(q1.checkbox('🍺 Alcool',   value=bool(pv('alc', True))))
        caf_bin   = int(q2.checkbox('☕ Caféine',  value=bool(pv('caf', True))))
        cho_bin   = int(q3.checkbox('🍫 Chocolat', value=bool(pv('cho', True))))
        nic_bin   = int(q4.checkbox('🚬 Nicotine', value=bool(pv('nic', False))))

        submitted = st.form_submit_button("🔍  Analyser et enregistrer le profil", use_container_width=True)

    if submitted:
        if not prenom_in.strip() or not nom_in.strip():
            st.error("Veuillez renseigner le prénom et le nom du patient avant d'analyser.")
        else:
            X_in = np.array([[
                AGE_ENC[age], GEN_ENC[gender], EDU_ENC[education],
                PAY_ENC[country], ETH_ENC[ethnicity],
                nscore, escore, oscore, ascore, cscore, impulsive, ss,
                alc_bin, caf_bin, cho_bin, nic_bin
            ]])

            results, all_scores = compute_predictions(X_in)
            verdict  = results[0]
            sc       = verdict['score']
            cls_v, icon_v, txt_v, badge_cls = verdict_style(sc)
            info     = SUB_INFO.get(verdict['sub'], {'emoji':'💊','cat':'Substance'})

            profile = dict(age=age, gender=gender, education=education,
                           country=country, ethnicity=ethnicity,
                           nscore=nscore, escore=escore, oscore=oscore,
                           ascore=ascore, cscore=cscore, impulsive=impulsive,
                           ss=ss, alc=alc_bin, caf=caf_bin, cho=cho_bin, nic=nic_bin)

            # Sauvegarde JSON
            entry = save_patient(prenom_in.strip(), nom_in.strip(),
                                 profile, all_scores,
                                 verdict['sub'], sc)
            st.session_state['loaded_patient'] = entry

            # Session
            pid = f"{prenom_in.strip()} {nom_in.strip()}"
            st.session_state['X_input']    = X_in
            st.session_state['results']    = results
            st.session_state['all_scores'] = all_scores
            st.session_state['history'].append({
                'name': pid, 'X': X_in, 'results': results,
                'all_scores': all_scores,
                'verdict_sub': verdict['sub'], 'verdict_score': sc,
                'profile': profile,
            })

            # ── Verdict
            st.markdown(f"""
            <div class='{cls_v}' style='margin-top:20px;'>
                <div style='display:flex;align-items:center;gap:16px;'>
                    <div style='font-size:44px;'>{info['emoji']}</div>
                    <div style='flex:1;'>
                        <div style='font-size:11px;color:#888;font-family:IBM Plex Mono,monospace;'>{info['cat']} · {pid}</div>
                        <div style='font-size:24px;font-weight:700;color:#1a2744;margin:4px 0;'>
                            {SUB_LABELS.get(verdict['sub'],verdict['sub'])}
                            <span class='{badge_cls}' style='font-size:12px;margin-left:8px;'>{txt_v}</span>
                        </div>
                        <div style='font-size:13px;color:#555;'>
                            {icon_v} Score : <strong>{sc*100:.1f}%</strong> ·
                            Facteur clé : <strong>{FEAT_LABELS.get(verdict['top_f'],verdict['top_f'])}</strong>
                            (β={verdict['coef']:+.2f})
                        </div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:48px;font-weight:700;'>{sc*100:.0f}<span style='font-size:22px;'>%</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top 3
            st.markdown("<br>", unsafe_allow_html=True)
            cols_r = st.columns(3)
            for i, r in enumerate(results[:3]):
                info_r = SUB_INFO.get(r['sub'], {'emoji':'💊','cat':''})
                gc     = '#ff6b6b' if r['score']>=0.60 else '#ffa94d' if r['score']>=0.40 else '#51cf66'
                with cols_r[i]:
                    st.markdown(f"""
                    <div class='card'>
                        <div style='font-size:13px;font-weight:600;color:#1a2744;margin-bottom:6px;'>
                            #{i+1} {info_r['emoji']} {SUB_LABELS.get(r['sub'],r['sub'])}
                        </div>
                        <div style='font-size:11px;color:#9aa5c0;margin-bottom:6px;'>MC:{r['pmc']*100:.0f}% · LR:{r['pbin']*100:.0f}%</div>
                        <div class='gauge-wrap'><div class='gauge-fill' style='width:{r["score"]*100:.0f}%;background:{gc};'></div></div>
                        <div style='font-size:20px;font-weight:700;color:{gc};'>{r["score"]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

            st.success(f"✅ Profil de **{pid}** enregistré dans `{DB_PATH}` — {len(entry['analyses'])} analyse(s) au total")
            st.markdown("<div class='disclaimer'>⚠️ Outil académique et préventif. Ne constitue pas un diagnostic médical.</div>", unsafe_allow_html=True)

    elif 'results' not in st.session_state:
        st.info("👆 Renseignez le prénom et le nom du patient, puis remplissez le formulaire.")

# ═══════════════════════════════════════════
#  TAB 2 — ANALYSE DÉTAILLÉE
# ═══════════════════════════════════════════
with tab2:
    if 'results' not in st.session_state:
        st.info("Remplissez d'abord le formulaire.")
    else:
        results = st.session_state['results']
        X_in    = st.session_state['X_input']

        sub_choice = st.selectbox(
            'Substance à analyser',
            [r['sub'] for r in results],
            format_func=lambda s: f"{SUB_INFO.get(s,{}).get('emoji','💊')}  {SUB_LABELS.get(s,s)}",
        )
        r_sel = next(r for r in results if r['sub'] == sub_choice)
        met   = metrics_bin[sub_choice]
        coefs = np.array(met['coefficients'])
        sc    = r_sel['score']

        m1,m2,m3,m4 = st.columns(4)
        for col, lbl, val, fmt in [
            (m1,'Score combiné',    sc,           '{:.1%}'),
            (m2,'Multi-Classe',     r_sel['pmc'], '{:.1%}'),
            (m3,'Régr. Logistique', r_sel['pbin'],'{:.1%}'),
            (m4,'F1 du modèle',     met['f1'],    '{:.3f}'),
        ]:
            with col:
                st.markdown(f"""<div class='metric-box'>
                    <div class='metric-val'>{fmt.format(val)}</div>
                    <div class='metric-lbl'>{lbl}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([1,1])

        with col_l:
            st.markdown("**Contributions des features pour ce patient**")
            st.caption("β × valeur_feature — bleu = augmente le risque · rouge = réduit")
            contribs = coefs * X_in[0]
            idx_s    = np.argsort(contribs)
            fig, ax  = plt.subplots(figsize=(6.5, 5.5))
            fig.patch.set_facecolor('white')
            colors_c = ['#4dabf7' if contribs[i]>0 else '#ff8787' for i in idx_s]
            ax.barh([FEAT_LABELS.get(FEATURE_COLS[i],FEATURE_COLS[i]) for i in idx_s],
                    contribs[idx_s], color=colors_c, edgecolor='white')
            ax.axvline(0, color='#888', linewidth=0.8)
            ax.set_xlabel('Contribution (β × valeur)', fontsize=9)
            ax.set_title(f'Décomposition — {SUB_LABELS.get(sub_choice,sub_choice)}',
                         fontsize=10, fontweight='bold', color='#1a2744')
            ax.set_facecolor('#f7f8fc')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=8.5)
            for i_idx, i_orig in enumerate(idx_s):
                v2 = contribs[i_orig]
                ax.text(v2+(0.005 if v2>=0 else -0.005), i_idx, f'{v2:+.3f}',
                        va='center', fontsize=7.5, ha='left' if v2>=0 else 'right', color='#333')
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col_r:
            st.markdown("**Top 3 facteurs — Interprétation clinique**")
            top3_feat = np.argsort(np.abs(contribs))[::-1][:3]
            for rank, idx in enumerate(top3_feat, 1):
                feat    = FEATURE_COLS[idx]
                coef    = coefs[idx]
                contrib = contribs[idx]
                title, desc = INTERP.get(feat, (FEAT_LABELS.get(feat,feat), ''))
                pos = contrib > 0
                st.markdown(f"""
                <div style='background:{"#fff8f8" if pos else "#f8fff9"};
                            border-left:3px solid {"#ff6b6b" if pos else "#51cf66"};
                            border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:10px;'>
                    <div style='font-size:12px;font-weight:600;color:#1a2744;'>#{rank} {title}</div>
                    <div style='font-size:11px;color:#666;margin:3px 0;'>{desc}</div>
                    <div style='font-size:11px;color:#888;'>
                        {"🔴 Risque" if pos else "🟢 Protecteur"} · β={coef:+.3f} · Contrib: <strong>{contrib:+.3f}</strong>
                    </div>
                </div>""", unsafe_allow_html=True)

            if sc >= 0.60:   reco="Entretien approfondi et orientation addictologue recommandés."; rc='#ff6b6b'
            elif sc >= 0.40: reco="Point de prévention conseillé avec supports adaptés."; rc='#ffa94d'
            else:            reco="Faible risque. Facteurs protecteurs actifs."; rc='#51cf66'
            st.markdown(f"""<div style='background:white;border:1px solid {rc}44;border-radius:10px;
                padding:14px 16px;margin-top:8px;'>
                <div style='font-size:11px;color:{rc};font-weight:600;margin-bottom:6px;'>RECOMMANDATION</div>
                <div style='font-size:12px;color:#555;line-height:1.6;'>{reco}</div></div>""",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  TAB 3 — EXPLICABILITÉ
# ═══════════════════════════════════════════
with tab3:
    st.markdown("### Coefficients β & Odds Ratios")
    exp1, exp2 = st.columns([1,1])

    with exp1:
        sub_exp   = st.selectbox('Substance', TARGET_IL,
                                 format_func=lambda s: f"{SUB_INFO.get(s,{}).get('emoji','💊')} {SUB_LABELS.get(s,s)}",
                                 key='exp_sub')
        coefs_exp = np.array(metrics_bin[sub_exp]['coefficients'])
        odds      = np.exp(coefs_exp)
        idx_s     = np.argsort(coefs_exp)
        labels_s  = [FEAT_LABELS.get(FEATURE_COLS[i],FEATURE_COLS[i]) for i in idx_s]

        st.markdown("**Coefficients β**")
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.patch.set_facecolor('white')
        colors_b = ['#63b3ed' if coefs_exp[i]>0 else '#fc8181' for i in idx_s]
        ax.barh(labels_s, coefs_exp[idx_s], color=colors_b, edgecolor='white')
        ax.axvline(0, color='#888', linewidth=0.8)
        ax.set_xlabel('Coefficient β', fontsize=9)
        ax.set_title(f'Coefficients — {SUB_LABELS.get(sub_exp,sub_exp)}',
                     fontsize=10, fontweight='bold', color='#1a2744')
        ax.set_facecolor('#f7f8fc')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8.5)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with exp2:
        st.markdown("**Odds Ratios** — OR > 1 = risque · OR < 1 = protecteur")
        odds_sorted = odds[idx_s]
        fig2, ax2   = plt.subplots(figsize=(6.5, 5.5))
        fig2.patch.set_facecolor('white')
        ax2.barh(labels_s, odds_sorted, color=['#fc8181' if o>1 else '#68d391' for o in odds_sorted], edgecolor='white')
        ax2.axvline(1.0, color='#333', linewidth=1.5, linestyle='--', alpha=0.6)
        ax2.set_xlabel('Odds Ratio', fontsize=9)
        ax2.set_title(f'Odds Ratios — {SUB_LABELS.get(sub_exp,sub_exp)}',
                      fontsize=10, fontweight='bold', color='#1a2744')
        ax2.set_facecolor('#f7f8fc')
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        ax2.tick_params(labelsize=8.5)
        for i, (label, o) in enumerate(zip(labels_s, odds_sorted)):
            ax2.text(o+0.01, i, f'{o:.2f}x', va='center', fontsize=7.5, color='#333')
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()

        rows_or = [{'Feature': FEAT_LABELS.get(FEATURE_COLS[i],FEATURE_COLS[i]),
                    'OR': round(odds[i],3),
                    'Effet': '⬆ Risque' if odds[i]>1 else '⬇ Protecteur'}
                   for i in np.argsort(odds)[::-1] if odds[i]>1.15 or odds[i]<0.87]
        if rows_or:
            st.dataframe(pd.DataFrame(rows_or), hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════
#  TAB 4 — HISTORIQUE & MATRICE
# ═══════════════════════════════════════════
with tab4:
    history = st.session_state['history']
    st.markdown("### Historique des patients & Matrice comparative")
    st.caption("Chaque ligne = un patient · Chaque colonne = une substance · Colonne bleue = moyenne des patients sélectionnés")

    if len(history) == 0:
        st.info("Aucun patient analysé en session. Remplissez le formulaire pour commencer.")
    else:
        patient_names = [h['name'] for h in history]
        col_sel, col_btn = st.columns([4, 1])
        with col_sel:
            selected_patients = st.multiselect('Patients à inclure dans la matrice',
                                               patient_names, default=patient_names)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Vider session"):
                st.session_state['history'] = []
                st.rerun()

        if not selected_patients:
            st.warning("Sélectionnez au moins un patient.")
        else:
            selected_hist = [h for h in history if h['name'] in selected_patients]
            avg_scores    = {sub: np.mean([h['all_scores'].get(sub,0.0) for h in selected_hist])
                             for sub in TARGET_IL}

            # ── Matrice HTML
            html = "<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;font-size:11px;'><thead><tr>"
            html += "<th style='background:#1a2744;color:white;padding:8px 10px;text-align:left;border:1px solid #2d3f60;min-width:120px;'>Patient</th>"
            html += "<th style='background:#1a2744;color:white;padding:8px 6px;text-align:center;border:1px solid #2d3f60;min-width:90px;'>Verdict</th>"
            for sub in TARGET_IL:
                emoji = SUB_INFO.get(sub,{}).get('emoji','💊')
                html += f"<th style='background:#1a2744;color:white;padding:6px 4px;text-align:center;border:1px solid #2d3f60;min-width:68px;'>{emoji}<br><span style='font-size:9px;'>{SUB_LABELS.get(sub,sub)}</span></th>"
            html += "<th style='background:#1e3a5f;color:#93C5FD;padding:6px 8px;text-align:center;border:2px solid #3B82F6;min-width:80px;'>📊<br><span style='font-size:9px;'>Moy. sélection</span></th>"
            html += "</tr></thead><tbody>"

            for i, h in enumerate(selected_hist):
                bg = '#ffffff' if i%2==0 else '#f8fafc'
                html += f"<tr style='background:{bg};'>"
                html += f"<td style='padding:7px 10px;font-weight:600;color:#1a2744;border:1px solid #e8ecf4;'>{h['name']}</td>"
                vs    = h['verdict_sub']; vsc = h['verdict_score']
                vc, _ = score_color(vsc)
                html += f"<td style='padding:7px 6px;text-align:center;border:1px solid #e8ecf4;font-size:11px;color:{vc};font-weight:600;'>{SUB_INFO.get(vs,{}).get('emoji','')} {SUB_LABELS.get(vs,vs)}<br><span style='font-size:10px;'>{vsc*100:.0f}%</span></td>"
                for sub in TARGET_IL:
                    sv      = h['all_scores'].get(sub,0.0)
                    tc, bgc = score_color(sv)
                    is_top  = sub == h['verdict_sub']
                    bdr     = f'border:2px solid {tc};' if is_top else 'border:1px solid #e8ecf4;'
                    star    = ' ★' if is_top else ''
                    html   += f"<td style='padding:6px 4px;text-align:center;background:{bgc};{bdr}'><span style='color:{tc};font-weight:{"700" if is_top else "500"};font-size:12px;'>{sv*100:.0f}%{star}</span></td>"
                html += "</tr>"

            # Ligne moyenne
            html += "<tr style='background:#EFF6FF;border-top:2px solid #3B82F6;'>"
            html += "<td style='padding:8px 10px;font-weight:700;color:#1e40af;border:1px solid #BFDBFE;font-size:12px;'>📊 Moyenne sélection</td>"
            html += "<td style='border:1px solid #BFDBFE;'></td>"
            for sub in TARGET_IL:
                avg     = avg_scores[sub]
                tc, bgc = score_color(avg)
                html   += f"<td style='padding:7px 4px;text-align:center;background:{bgc};border:1px solid #BFDBFE;'><span style='color:{tc};font-weight:700;font-size:13px;'>{avg*100:.0f}%</span></td>"
            grand_avg   = np.mean(list(avg_scores.values()))
            tc_g, bgc_g = score_color(grand_avg)
            html += f"<td style='padding:7px 8px;text-align:center;background:#dbeafe;border:2px solid #3B82F6;'><span style='color:#1d4ed8;font-weight:700;font-size:14px;'>{grand_avg*100:.0f}%</span></td>"
            html += "</tr></tbody></table></div>"
            st.markdown(html, unsafe_allow_html=True)

            st.markdown("""<div style='display:flex;gap:20px;margin-top:10px;font-size:11px;color:#555;flex-wrap:wrap;'>
                <span>🔴 ≥60%</span><span>🟠 40–60%</span><span>🟢 <40%</span>
                <span>★ Substance principale</span>
                <span style='color:#1d4ed8;font-weight:600;'>📊 Colonne bleue = moyenne de la sélection</span>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            # Graphique moyenne
            avg_sorted = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            fig_avg, ax_avg = plt.subplots(figsize=(11, 4))
            fig_avg.patch.set_facecolor('white')
            xlabels = [f"{SUB_INFO.get(s,{}).get('emoji','')} {SUB_LABELS.get(s,s)}" for s,_ in avg_sorted]
            yvals   = [v*100 for _,v in avg_sorted]
            colors  = ['#ff6b6b' if v>=60 else '#ffa94d' if v>=40 else '#51cf66' for v in yvals]
            bars    = ax_avg.bar(xlabels, yvals, color=colors, edgecolor='white')
            ax_avg.axhline(60, color='#ff6b6b', linestyle='--', linewidth=1, alpha=0.4)
            ax_avg.axhline(40, color='#ffa94d', linestyle='--', linewidth=1, alpha=0.4)
            ax_avg.set_ylim(0,105); ax_avg.set_ylabel('Score moyen (%)', fontsize=10)
            ax_avg.set_title(f'Profil de risque moyen — {len(selected_hist)} patient(s)',
                             fontsize=11, fontweight='bold', color='#1a2744')
            ax_avg.set_facecolor('#f7f8fc')
            ax_avg.tick_params(axis='x', labelrotation=30, labelsize=9)
            ax_avg.spines['top'].set_visible(False); ax_avg.spines['right'].set_visible(False)
            for bar, val in zip(bars, yvals):
                ax_avg.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                            f'{val:.0f}%', ha='center', fontsize=8.5, color='#333')
            plt.tight_layout(); st.pyplot(fig_avg, use_container_width=True); plt.close()

            if len(selected_hist) > 1:
                n_high = sum(1 for h in selected_hist if h['verdict_score']>=0.60)
                n_med  = sum(1 for h in selected_hist if 0.40<=h['verdict_score']<0.60)
                n_low  = sum(1 for h in selected_hist if h['verdict_score']<0.40)
                top_s  = max(avg_scores, key=avg_scores.get)
                cs1,cs2,cs3,cs4 = st.columns(4)
                cs1.metric("Risque élevé",  f"{n_high}/{len(selected_hist)}")
                cs2.metric("Risque modéré", f"{n_med}/{len(selected_hist)}")
                cs3.metric("Faible risque", f"{n_low}/{len(selected_hist)}")
                cs4.metric("Substance + risquée", f"{SUB_INFO.get(top_s,{}).get('emoji','')} {SUB_LABELS.get(top_s,top_s)}")

# ═══════════════════════════════════════════
#  TAB 5 — PROFILS POPULATION
# ═══════════════════════════════════════════
with tab5:
    st.markdown("### Profils psychologiques par substance")
    PROFILES = {
        'cannabis':  [0.10,-0.05, 0.50,-0.15,-0.30, 0.40, 0.80],
        'ecstasy':   [0.20, 0.40, 0.70,-0.30,-0.50, 0.70, 1.20],
        'lsd':       [0.15, 0.10, 1.10,-0.20,-0.40, 0.50, 1.00],
        'heroin':    [0.80,-0.30, 0.20,-0.50,-0.80, 1.00, 0.60],
        'meth':      [0.60,-0.10, 0.30,-0.40,-0.70, 0.90, 0.90],
        'ketamine':  [0.30, 0.20, 0.80,-0.25,-0.45, 0.65, 1.10],
        'coke':      [0.25, 0.50, 0.40,-0.20,-0.30, 0.70, 0.85],
        'amphet':    [0.20, 0.30, 0.50,-0.10,-0.40, 0.75, 0.95],
        'mushrooms': [0.10, 0.00, 1.20,-0.10,-0.20, 0.40, 0.90],
        'benzos':    [0.90,-0.40,-0.10,-0.30,-0.50, 0.50, 0.30],
        'legalh':    [0.20, 0.20, 0.60,-0.15,-0.35, 0.55, 0.80],
        'amyl':      [0.15, 0.60, 0.50,-0.10,-0.25, 0.60, 1.00],
        'vsa':       [0.40,-0.20, 0.20,-0.35,-0.60, 0.80, 0.70],
    }
    PSYCH_LBLS = ['Neuroticisme','Extraversion','Ouverture','Agréabilité','Conscienciosité','Impulsivité','SS']

    selected_subs = st.multiselect('Comparer les substances', TARGET_IL,
                                   default=['cannabis','ecstasy','heroin','lsd'],
                                   format_func=lambda s: f"{SUB_INFO.get(s,{}).get('emoji','💊')} {SUB_LABELS.get(s,s)}")
    if selected_subs:
        cr, ch = st.columns([1,1])
        with cr:
            st.markdown("**Radar**")
            N = len(PSYCH_LBLS); angles = [n/N*2*np.pi for n in range(N)] + [0]
            pal = plt.cm.Set2(np.linspace(0,1,len(selected_subs)))
            fig_r, ax_r = plt.subplots(figsize=(5.5,5.5), subplot_kw=dict(polar=True))
            fig_r.patch.set_facecolor('white'); ax_r.set_facecolor('#f7f8fc')
            for i, sub in enumerate(selected_subs):
                vn = [(v+3)/6 for v in PROFILES[sub]] + [(PROFILES[sub][0]+3)/6]
                ax_r.plot(angles, vn, 'o-', linewidth=2, color=pal[i], label=SUB_LABELS.get(sub,sub))
                ax_r.fill(angles, vn, alpha=0.08, color=pal[i])
            ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(PSYCH_LBLS, fontsize=8.5)
            ax_r.set_ylim(0,1); ax_r.set_yticks([]); ax_r.legend(loc='upper right', bbox_to_anchor=(1.35,1.1), frameon=False, fontsize=8)
            ax_r.grid(color='#CCCCCC', linewidth=0.6); ax_r.spines['polar'].set_color('#CCCCCC')
            plt.tight_layout(); st.pyplot(fig_r, use_container_width=True); plt.close()

        with ch:
            st.markdown("**Heatmap**")
            data_h = pd.DataFrame({SUB_LABELS.get(s,s): PROFILES[s] for s in selected_subs}, index=PSYCH_LBLS)
            fig_h, ax_h = plt.subplots(figsize=(5.5,5))
            fig_h.patch.set_facecolor('white')
            im = ax_h.imshow(data_h.values, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
            plt.colorbar(im, ax=ax_h, label='Score moyen (z-score)')
            ax_h.set_xticks(range(len(selected_subs)))
            ax_h.set_xticklabels([SUB_LABELS.get(s,s) for s in selected_subs], rotation=30, ha='right', fontsize=9)
            ax_h.set_yticks(range(len(PSYCH_LBLS))); ax_h.set_yticklabels(PSYCH_LBLS, fontsize=9)
            for i in range(len(PSYCH_LBLS)):
                for j, sub in enumerate(selected_subs):
                    v = PROFILES[sub][i]
                    ax_h.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8, color='white' if abs(v)>0.8 else 'black')
            plt.tight_layout(); st.pyplot(fig_h, use_container_width=True); plt.close()

# ═══════════════════════════════════════════
#  TAB 6 — MÉTHODOLOGIE
# ═══════════════════════════════════════════
with tab6:
    st.markdown("### Méthodologie & Limites")
    m1c, m2c = st.columns([1,1])
    with m1c:
        st.markdown("""<div class='card'><div class='card-title'>Pipeline ML</div>
        <div style='font-size:13px;color:#444;line-height:1.8;'>
        <strong>Étape 1 — Multi-Classe (Random Forest)</strong><br>F1 macro : <strong>0.90</strong><br><br>
        <strong>Étape 2 — Binaire (Régression Logistique)</strong><br>F1 moyen : <strong>0.79</strong><br><br>
        <strong>Score combiné</strong> : 40% MC + 60% LR
        </div></div>""", unsafe_allow_html=True)
        perf = [{'Substance': f"{SUB_INFO.get(s,{}).get('emoji','')} {SUB_LABELS.get(s,s)}",
                 'F1': round(metrics_bin[s]['f1'],3),
                 'Recall': round(metrics_bin[s]['recall'],3),
                 'AUC': round(metrics_bin[s]['roc_auc'],3)} for s in TARGET_IL]
        st.dataframe(pd.DataFrame(perf), hide_index=True, use_container_width=True)
    with m2c:
        st.markdown("""<div class='card'><div class='card-title'>Système de persistance JSON</div>
        <div style='font-size:13px;color:#444;line-height:1.8;'>
        Les profils patients sont enregistrés dans <code>patients_db.json</code>.<br><br>
        À chaque nouvelle analyse, les paramètres (âge, personnalité, consommations légales) et les scores par substance sont sauvegardés.<br><br>
        Lors d'une consultation ultérieure, la saisie du prénom + nom <strong>recharge automatiquement</strong> le profil et pré-remplit tous les sliders et menus.<br><br>
        L'historique des analyses successives est conservé pour suivre l'évolution dans le temps.
        </div></div>""", unsafe_allow_html=True)
        st.markdown("""<div class='card'><div class='card-title'>Limites</div>
        <div style='font-size:13px;color:#444;line-height:1.8;'>
        ⚠️ Données auto-déclarées · Population non représentative<br>
        ⚠️ Variables manquantes (socio-éco, traumatismes)<br>
        ⚠️ Outil préventif ≠ diagnostic clinique validé
        </div></div>""", unsafe_allow_html=True)
