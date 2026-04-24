# IMPORTANT : importer pytorch_tabnet AVANT xgboost (conflit DLL Windows)
from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="LoL Pro Match Predictor", layout="wide")

CHECKPOINTS = ['10min', '15min', '20min', '25min']
CP_LABELS   = {'10min': '10 min', '15min': '15 min', '20min': '20 min', '25min': '25 min'}

# ─── Chargement modèles ──────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    models = {}
    for cp in CHECKPOINTS:
        with open(f'models/xgboost/xgb_{cp}.pkl', 'rb') as f:
            xgb = pickle.load(f)
        with open(f'models/preprocessing/scaler_{cp}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'models/preprocessing/features_{cp}.pkl', 'rb') as f:
            feats = pickle.load(f)
        with open(f'models/tabnet/tabnet_importances_{cp}.pkl', 'rb') as f:
            importances = pickle.load(f)
        tabnet = TabNetClassifier()
        tabnet.load_model(f'models/tabnet/tabnet_{cp}.zip')
        ftt = torch.load(f'models/ftt/ftt_{cp}.pt', map_location='cpu', weights_only=False)
        ftt.eval()
        models[cp] = {
            'xgb': xgb, 'tabnet': tabnet, 'ftt': ftt,
            'scaler': scaler, 'features': feats, 'importances': importances,
        }
    return models

@st.cache_data
def load_dataset():
    df = pd.read_csv('data/2025_LoL_esports_match_data_from_OraclesElixir.csv', low_memory=False)
    df = df[
        (df['participantid'].isin([100, 200])) &
        (df['datacompleteness'] == 'complete')
    ].copy()
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

all_models = load_all_models()
df_raw     = load_dataset()

# ─── Helpers ─────────────────────────────────────────────────────────────────
def build_stats(row, is_blue, checkpoint):
    sign = 1 if is_blue else -1
    def v(col): return float(row[col]) if col in row.index and pd.notna(row.get(col, np.nan)) else 0.0

    stats = {
        'side':         1 if is_blue else 0,
        'firstblood':   v('firstblood'),
        'golddiffat10': sign * v('golddiffat10'),
        'xpdiffat10':   sign * v('xpdiffat10'),
        'csdiffat10':   sign * v('csdiffat10'),
        'killsat10':    v('killsat10'),
        'assistsat10':  v('assistsat10'),
        'deathsat10':   v('deathsat10'),
    }
    if checkpoint in ('15min', '20min', '25min'):
        stats.update({
            'firstdragon':  v('firstdragon'),
            'firstherald':  v('firstherald'),
            'firsttower':   v('firsttower'),
            'golddiffat15': sign * v('golddiffat15'),
            'xpdiffat15':   sign * v('xpdiffat15'),
            'csdiffat15':   sign * v('csdiffat15'),
            'killsat15':    v('killsat15'),
            'assistsat15':  v('assistsat15'),
            'deathsat15':   v('deathsat15'),
        })
    if checkpoint in ('20min', '25min'):
        stats.update({
            'golddiffat20': sign * v('golddiffat20'),
            'xpdiffat20':   sign * v('xpdiffat20'),
            'csdiffat20':   sign * v('csdiffat20'),
            'killsat20':    v('killsat20'),
            'assistsat20':  v('assistsat20'),
            'deathsat20':   v('deathsat20'),
        })
    if checkpoint == '25min':
        stats.update({
            'firstbaron':   v('firstbaron'),
            'golddiffat25': sign * v('golddiffat25'),
            'xpdiffat25':   sign * v('xpdiffat25'),
            'csdiffat25':   sign * v('csdiffat25'),
            'killsat25':    v('killsat25'),
            'assistsat25':  v('assistsat25'),
            'deathsat25':   v('deathsat25'),
        })
    return stats

def predict_at(stats_dict, checkpoint):
    """Retourne (p_xgb, p_tab, p_ftt)."""
    m   = all_models[checkpoint]
    X   = np.array([[stats_dict[f] for f in m['features']]])
    X_s = m['scaler'].transform(X)
    p_xgb = m['xgb'].predict_proba(X)[0][1]
    p_tab = m['tabnet'].predict_proba(X_s)[0][1]
    with torch.no_grad():
        logit = m['ftt'](torch.FloatTensor(X_s), None).squeeze()
        p_ftt = torch.sigmoid(logit).item()
    return p_xgb, p_tab, p_ftt

def get_local_importance(stats_dict, checkpoint):
    """Masques d'attention TabNet pour une prédiction individuelle."""
    m   = all_models[checkpoint]
    X   = np.array([[stats_dict[f] for f in m['features']]])
    X_s = m['scaler'].transform(X)
    M_explain, _ = m['tabnet'].explain(X_s)
    return M_explain[0]

def feat_colors(feat_names):
    return [
        '#f1c40f' if 'gold' in f else
        '#9b59b6' if 'xp' in f else
        '#2ecc71' if 'cs' in f else
        '#e74c3c' if 'kill' in f or 'death' in f else
        '#3498db' for f in feat_names
    ]

LEGEND_PATCHES = [
    mpatches.Patch(color='#f1c40f', label='Gold'),
    mpatches.Patch(color='#9b59b6', label='XP'),
    mpatches.Patch(color='#2ecc71', label='CS'),
    mpatches.Patch(color='#e74c3c', label='Kills / Deaths'),
    mpatches.Patch(color='#3498db', label='Objectifs / Contexte'),
]

# ─── Render helpers ───────────────────────────────────────────────────────────
def render_timeline_2teams(probs_blue, probs_red, blue_team, red_team, checkpoints_used):
    labels = [CP_LABELS[cp] for cp in checkpoints_used]

    def normalize(a, b): return a / (a + b), b / (a + b)

    tab_blue, tab_red = [], []
    xgb_blue, xgb_red = [], []
    ftt_blue, ftt_red = [], []

    for cp in checkpoints_used:
        nb, nr = normalize(probs_blue[cp][1], probs_red[cp][1])
        xb, xr = normalize(probs_blue[cp][0], probs_red[cp][0])
        fb, fr = normalize(probs_blue[cp][2], probs_red[cp][2])
        tab_blue.append(nb); tab_red.append(nr)
        xgb_blue.append(xb); xgb_red.append(xr)
        ftt_blue.append(fb); ftt_red.append(fr)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(labels, tab_blue, 'o-',  color='#1a75ff', lw=2.5, ms=9, label=f'{blue_team} - TabNet',          zorder=5)
    ax.plot(labels, ftt_blue, '^:',  color='#1a75ff', lw=2.0, ms=8, label=f'{blue_team} - FT-Transformer',  zorder=4, alpha=0.85)
    ax.plot(labels, xgb_blue, 'o--', color='#1a75ff', lw=1.5, ms=6, label=f'{blue_team} - XGBoost',         alpha=0.4)
    ax.plot(labels, tab_red,  's-',  color='#cc0000', lw=2.5, ms=9, label=f'{red_team} - TabNet',            zorder=5)
    ax.plot(labels, ftt_red,  'v:',  color='#cc0000', lw=2.0, ms=8, label=f'{red_team} - FT-Transformer',   zorder=4, alpha=0.85)
    ax.plot(labels, xgb_red,  's--', color='#cc0000', lw=1.5, ms=6, label=f'{red_team} - XGBoost',          alpha=0.4)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6, lw=1.2)
    ax.fill_between(labels, 0.5, tab_blue, alpha=0.07, color='#1a75ff')
    ax.fill_between(labels, 0.5, tab_red,  alpha=0.07, color='#cc0000')
    ax.set_ylim(0, 1)
    ax.set_ylabel('P(victoire) normalisée')
    ax.set_title('Évolution de la prédiction au cours du match (normalisée - somme à 100%)')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    for val, color in [(tab_blue[-1], '#1a75ff'), (tab_red[-1], '#cc0000')]:
        ax.annotate(f'{val:.1%}', (labels[-1], val), xytext=(8, 0),
                    textcoords='offset points', color=color, fontsize=10, va='center', fontweight='bold')
    plt.tight_layout()
    return fig

def render_timeline_1team(probs, checkpoints_used):
    labels   = [CP_LABELS[cp] for cp in checkpoints_used]
    tab_vals = [probs[cp][1] for cp in checkpoints_used]
    xgb_vals = [probs[cp][0] for cp in checkpoints_used]
    ftt_vals = [probs[cp][2] for cp in checkpoints_used]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(labels, tab_vals, 'o-',  color='#2ecc71', lw=2.5, ms=9, label='TabNet',         zorder=5)
    ax.plot(labels, ftt_vals, '^:',  color='#27ae60', lw=2.0, ms=8, label='FT-Transformer', zorder=4)
    ax.plot(labels, xgb_vals, 'o--', color='#2ecc71', lw=1.5, ms=6, label='XGBoost',        alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6, lw=1.2)
    ax.fill_between(labels, 0.5, tab_vals, alpha=0.1, color='#2ecc71')
    ax.set_ylim(0, 1)
    ax.set_ylabel('P(victoire)')
    ax.set_title('Évolution de la prédiction')
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    plt.tight_layout()
    return fig

def render_importance_bar(importances, feat_names, title):
    s = pd.Series(importances, index=feat_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, len(feat_names) * 0.38)))
    s.plot(kind='barh', ax=ax, color=feat_colors(s.index), edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Attention (contribution)')
    ax.legend(handles=LEGEND_PATCHES, loc='lower right', fontsize=8)
    plt.tight_layout()
    return fig

def render_team_card(col, team_name, side_label, prob_xgb, prob_tab, prob_ftt, actual_result):
    with col:
        won        = actual_result == 1
        side_color = "#1043a0" if "Blue" in side_label else "#c0392b"
        st.markdown(
            f"<div style='background:{side_color};padding:8px 14px;border-radius:8px;"
            f"text-align:center;color:white;font-weight:bold;font-size:1rem;margin-bottom:8px'>"
            f"{team_name} - {side_label}</div>", unsafe_allow_html=True
        )
        rc = "#1a6b34" if won else "#c0392b"
        rl = "Victoire (réel)" if won else "Défaite (réel)"
        st.markdown(
            f"<div style='background:{rc};padding:6px;border-radius:6px;"
            f"text-align:center;color:white;font-weight:bold;margin-bottom:12px'>{rl}</div>",
            unsafe_allow_html=True
        )
        for model_name, prob in [("XGBoost", prob_xgb), ("TabNet", prob_tab), ("FT-Transformer", prob_ftt)]:
            vc = "#2ecc71" if prob >= 0.6 else "#e74c3c" if prob <= 0.4 else "#f39c12"
            st.markdown(f"**{model_name}** - P(victoire) : `{prob:.1%}`")
            fig, ax = plt.subplots(figsize=(4, 0.4))
            ax.barh(0, prob,       color=vc,       height=0.5)
            ax.barh(0, 1 - prob,   left=prob, color="#ecf0f1", height=0.5)
            ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
            ax.set_xlim(0, 1); ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("LoL Pro Match Predictor")
st.markdown(
    "Prédiction de l'issue d'un match pro à **10, 15 et 20 min**.  \n"
    "Comparaison **XGBoost** (baseline) vs **TabNet** (Arik & Pfister, 2021) vs **FT-Transformer** (Gorishniy et al., NeurIPS 2021)."
)
st.divider()

# ─── Onglets ─────────────────────────────────────────────────────────────────
tab_load, tab_manual = st.tabs(["Charger un match existant", "Saisie manuelle"])

match_loaded = False

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - Charger un match existant
# ══════════════════════════════════════════════════════════════════════════════
with tab_load:
    col_a, col_b = st.columns(2)
    with col_a:
        leagues = sorted(df_raw['league'].unique())
        league  = st.selectbox("Ligue", leagues, index=leagues.index('LEC') if 'LEC' in leagues else 0)

    df_league = df_raw[df_raw['league'] == league]
    matches   = {}
    for gid in df_league['gameid'].unique():
        game_rows = df_league[df_league['gameid'] == gid]
        blue_row  = game_rows[game_rows['side'] == 'Blue']
        red_row   = game_rows[game_rows['side'] == 'Red']
        if len(blue_row) == 0 or len(red_row) == 0:
            continue
        label = f"{blue_row.iloc[0]['date']} - {blue_row.iloc[0]['teamname']} vs {red_row.iloc[0]['teamname']}"
        matches[label] = gid

    with col_b:
        if matches:
            selected_label = st.selectbox("Match", list(matches.keys()))
            selected_gid   = matches[selected_label]
        else:
            st.warning("Aucun match trouvé.")
            selected_gid = None

    if selected_gid:
        game_rows = df_league[df_league['gameid'] == selected_gid]
        blue_row  = game_rows[game_rows['side'] == 'Blue'].iloc[0]
        red_row   = game_rows[game_rows['side'] == 'Red'].iloc[0]
        blue_team = blue_row['teamname']
        red_team  = red_row['teamname']

        probs_blue, probs_red = {}, {}
        for cp in CHECKPOINTS:
            probs_blue[cp] = predict_at(build_stats(blue_row, True,  cp), cp)
            probs_red[cp]  = predict_at(build_stats(red_row,  False, cp), cp)

        # Stats summary @15 min
        st.markdown("**Stats @15 min**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Gold diff",          f"{blue_row.get('golddiffat15', 0):+.0f}", help=f"Positif = avantage {blue_team}")
        c2.metric(f"Kills {blue_team}", f"{blue_row.get('killsat15', 0):.0f}")
        c3.metric(f"Kills {red_team}",  f"{red_row.get('killsat15', 0):.0f}")
        c4.metric("First Dragon", "Blue" if blue_row.get('firstdragon', 0) == 1 else "Red")
        c5.metric("First Herald", "Blue" if blue_row.get('firstherald', 0) == 1 else "Red")
        c6.metric("First Tower",  "Blue" if blue_row.get('firsttower',  0) == 1 else "Red")

        st.divider()

        # ── Timeline ──────────────────────────────────────────────────────────
        st.subheader("Evolution de la prédiction")
        fig = render_timeline_2teams(probs_blue, probs_red, blue_team, red_team, CHECKPOINTS)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption(
            f"Graphique linéaire - évolution de P(victoire) normalisée pour {blue_team} (bleu) "
            f"et {red_team} (rouge) aux checkpoints 10, 15 et 20 min. "
            f"Trait plein = TabNet, pointillé-triangle = FT-Transformer, tirets = XGBoost. "
            f"La ligne horizontale marque le seuil 50 %."
        )

        st.divider()

        # ── Team cards - un onglet par checkpoint ────────────────────────────
        st.subheader("Prédictions par checkpoint")
        pred_tabs = st.tabs([f"@{CP_LABELS[cp]}" for cp in CHECKPOINTS])
        for pred_tab, cp in zip(pred_tabs, CHECKPOINTS):
            with pred_tab:
                col_blue, col_mid, col_red = st.columns([5, 1, 5])
                render_team_card(col_blue, blue_team, "Blue side",
                                 probs_blue[cp][0], probs_blue[cp][1], probs_blue[cp][2],
                                 int(blue_row['result']))
                with col_mid:
                    st.markdown("<div style='text-align:center;font-size:1.75rem;margin-top:60px'>VS</div>",
                                unsafe_allow_html=True)
                render_team_card(col_red, red_team, "Red side",
                                 probs_red[cp][0], probs_red[cp][1], probs_red[cp][2],
                                 int(red_row['result']))

        st.divider()

        # ── Interprétabilité ──────────────────────────────────────────────────
        st.subheader("Interprétabilité TabNet")
        tab_loc, tab_glob = st.tabs(["Ce match (locale)", "Globale (tous les matchs)"])

        with tab_loc:
            st.markdown("Quelles features ont **le plus contribué** à la prédiction de **ce match précis** ?")
            cp_loc = st.radio("Checkpoint", [f"@{CP_LABELS[cp]}" for cp in CHECKPOINTS],
                              horizontal=True, key="cp_loc")
            cp_key = {f"@{CP_LABELS[cp]}": cp for cp in CHECKPOINTS}[cp_loc]
            stats_b = build_stats(blue_row, True,  cp_key)
            stats_r = build_stats(red_row,  False, cp_key)
            local_b    = get_local_importance(stats_b, cp_key)
            local_r    = get_local_importance(stats_r, cp_key)
            feat_names = all_models[cp_key]['features']
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                fig = render_importance_bar(local_b, feat_names, f"{blue_team} - contribution locale")
                st.pyplot(fig, use_container_width=True); plt.close()
                st.caption(f"Barres horizontales - contribution de chaque variable à la prédiction TabNet pour {blue_team} sur ce match. Couleurs : jaune = gold, violet = XP, vert = CS, rouge = kills/deaths, bleu = objectifs.")
            with col_l2:
                fig = render_importance_bar(local_r, feat_names, f"{red_team} - contribution locale")
                st.pyplot(fig, use_container_width=True); plt.close()
                st.caption(f"Barres horizontales - contribution de chaque variable à la prédiction TabNet pour {red_team} sur ce match. Même code couleur que le graphique de gauche.")

        with tab_glob:
            st.markdown("Importance **moyenne** des features TabNet sur l'ensemble du jeu de test.")
            cp_glob = st.radio("Checkpoint", [f"@{CP_LABELS[cp]}" for cp in CHECKPOINTS],
                               horizontal=True, key="cp_glob")
            cp_key_g = {f"@{CP_LABELS[cp]}": cp for cp in CHECKPOINTS}[cp_glob]
            fig = render_importance_bar(
                all_models[cp_key_g]['importances'], all_models[cp_key_g]['features'],
                f"Masques d'attention TabNet (moyennés) - @{CP_LABELS[cp_key_g]}"
            )
            st.pyplot(fig, use_container_width=True); plt.close()
            st.caption(f"Barres horizontales - importance moyenne des variables pour le modèle TabNet @{CP_LABELS[cp_key_g]}, calculée sur l'ensemble du jeu de test. Une valeur plus élevée indique une variable plus déterminante.")

        match_loaded = True

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - Saisie manuelle
# ══════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown("Stats exprimées du **point de vue de ton équipe**. Positif = ton équipe est en avance.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("@10 minutes")
        fb_m         = st.toggle("First Blood",   key="fb_m")
        golddiff10_m = st.slider("Avantage gold", -7000,  7000,  0, 100, key="g10m")
        xpdiff10_m   = st.slider("Avantage XP",   -5000,  5000,  0, 100, key="x10m")
        csdiff10_m   = st.slider("Avantage CS",   -100,   100,   0, 1,   key="c10m")
        kills10_m    = st.number_input("Kills",   0, 20, 2, key="k10m")
        assists10_m  = st.number_input("Assists", 0, 35, 3, key="a10m")
        deaths10_m   = st.number_input("Deaths",  0, 20, 2, key="d10m")

    with col_m2:
        st.subheader("@15 minutes")
        fd_m         = st.toggle("Premier Dragon", key="fd_m")
        fh_m         = st.toggle("Premier Herald", key="fh_m")
        ft_m         = st.toggle("Première Tour",  key="ft_m")
        golddiff15_m = st.slider("Avantage gold", -13000, 13000, 0, 100, key="g15m")
        xpdiff15_m   = st.slider("Avantage XP",   -9000,  9000,  0, 100, key="x15m")
        csdiff15_m   = st.slider("Avantage CS",   -200,   200,   0, 1,   key="c15m")
        kills15_m    = st.number_input("Kills",   0, 30, 4, key="k15m")
        assists15_m  = st.number_input("Assists", 0, 50, 7, key="a15m")
        deaths15_m   = st.number_input("Deaths",  0, 30, 4, key="d15m")

    col_m3, col_m4 = st.columns(2)
    with col_m3:
        st.subheader("@20 minutes")
        golddiff20_m = st.slider("Avantage gold", -20000, 20000, 0, 100, key="g20m")
        xpdiff20_m   = st.slider("Avantage XP",   -14000, 14000, 0, 100, key="x20m")
        csdiff20_m   = st.slider("Avantage CS",   -300,   300,   0, 1,   key="c20m")
        kills20_m    = st.number_input("Kills",   0, 40, 6, key="k20m")
        assists20_m  = st.number_input("Assists", 0, 70, 10, key="a20m")
        deaths20_m   = st.number_input("Deaths",  0, 40, 6, key="d20m")

    with col_m4:
        st.subheader("@25 minutes")
        fb25_m       = st.toggle("Premier Baron", key="fb25_m")
        golddiff25_m = st.slider("Avantage gold", -28000, 28000, 0, 100, key="g25m")
        xpdiff25_m   = st.slider("Avantage XP",   -18000, 18000, 0, 100, key="x25m")
        csdiff25_m   = st.slider("Avantage CS",   -400,   400,   0, 1,   key="c25m")
        kills25_m    = st.number_input("Kills",   0, 50, 8, key="k25m")
        assists25_m  = st.number_input("Assists", 0, 90, 14, key="a25m")
        deaths25_m   = st.number_input("Deaths",  0, 50, 8, key="d25m")

    if st.button("Prédire", type="primary"):
        base10 = {
            'side': 1, 'firstblood': int(fb_m),
            'golddiffat10': golddiff10_m, 'xpdiffat10': xpdiff10_m, 'csdiffat10': csdiff10_m,
            'killsat10': kills10_m, 'assistsat10': assists10_m, 'deathsat10': deaths10_m,
        }
        base15 = {
            'firstdragon': int(fd_m), 'firstherald': int(fh_m), 'firsttower': int(ft_m),
            'golddiffat15': golddiff15_m, 'xpdiffat15': xpdiff15_m, 'csdiffat15': csdiff15_m,
            'killsat15': kills15_m, 'assistsat15': assists15_m, 'deathsat15': deaths15_m,
        }
        base20 = {
            'golddiffat20': golddiff20_m, 'xpdiffat20': xpdiff20_m, 'csdiffat20': csdiff20_m,
            'killsat20': kills20_m, 'assistsat20': assists20_m, 'deathsat20': deaths20_m,
        }
        st.session_state['manual_stats'] = {
            '10min': base10,
            '15min': {**base10, **base15},
            '20min': {**base10, **base15, **base20},
            '25min': {**base10, **base15, **base20,
                      'firstbaron': int(fb25_m),
                      'golddiffat25': golddiff25_m, 'xpdiffat25': xpdiff25_m, 'csdiffat25': csdiff25_m,
                      'killsat25': kills25_m, 'assistsat25': assists25_m, 'deathsat25': deaths25_m},
        }

manual_stats = st.session_state.get('manual_stats', None)

if manual_stats is not None:
    st.divider()

    probs_manual = {cp: predict_at(manual_stats[cp], cp) for cp in CHECKPOINTS}

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.subheader("Evolution de la prédiction")
    fig = render_timeline_1team(probs_manual, CHECKPOINTS)
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("Graphique linéaire - évolution de P(victoire) de ton équipe aux checkpoints 10 et 15 min selon les trois modèles. Trait plein = TabNet, pointillé-triangle = FT-Transformer, tirets = XGBoost. La ligne horizontale marque le seuil 50 %.")

    st.divider()

    MANUAL_CPS = ['10min', '15min', '20min', '25min']
    st.subheader("Prédictions par checkpoint")
    pred_tabs_m = st.tabs([f"@{CP_LABELS[cp]}" for cp in MANUAL_CPS])
    for pred_tab_m, cp in zip(pred_tabs_m, MANUAL_CPS):
        with pred_tab_m:
            p_xgb, p_tab, p_ftt = probs_manual[cp]
            col1, col2, col3 = st.columns(3)
            for col, name, prob in [(col1, "XGBoost", p_xgb), (col2, "TabNet", p_tab), (col3, "FT-Transformer", p_ftt)]:
                color = "#1a6b34" if prob >= 0.6 else "#c0392b" if prob <= 0.4 else "#7a5000"
                label = "Victoire probable" if prob >= 0.6 else "Défaite probable" if prob <= 0.4 else "Match serré"
                with col:
                    st.subheader(name)
                    st.metric("P(victoire de ton équipe)", f"{prob:.1%}")
                    st.markdown(
                        f"<div style='background:{color};padding:12px;border-radius:8px;"
                        f"text-align:center;font-size:1.125rem;font-weight:bold;color:white'>{label}</div>",
                        unsafe_allow_html=True
                    )

    st.divider()

    # ── Interprétabilité ──────────────────────────────────────────────────────
    st.subheader("Interprétabilité TabNet")
    tab_loc_m, tab_glob_m = st.tabs(["Cette prédiction (locale)", "Globale (tous les matchs)"])

    with tab_loc_m:
        st.markdown("Quelles features ont **le plus contribué** à **cette prédiction** ?")
        cp_loc_m = st.radio("Checkpoint", [f"@{CP_LABELS[cp]}" for cp in MANUAL_CPS],
                            horizontal=True, key="cp_loc_m")
        cp_key_m = {f"@{CP_LABELS[cp]}": cp for cp in MANUAL_CPS}[cp_loc_m]
        feat_names_m = all_models[cp_key_m]['features']
        local_imp    = get_local_importance(manual_stats[cp_key_m], cp_key_m)
        fig = render_importance_bar(local_imp, feat_names_m,
                                    f"Ton équipe - contribution locale @{CP_LABELS[cp_key_m]}")
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption(f"Barres horizontales - contribution de chaque variable à la prédiction TabNet pour cette saisie @{CP_LABELS[cp_key_m]}. Couleurs : jaune = gold, violet = XP, vert = CS, rouge = kills/deaths, bleu = objectifs.")

    with tab_glob_m:
        st.markdown("Importance **moyenne** des features TabNet sur l'ensemble du jeu de test.")
        cp_glob_m = st.radio("Checkpoint", [f"@{CP_LABELS[cp]}" for cp in CHECKPOINTS],
                             horizontal=True, key="cp_glob_m")
        cp_key_gm = {f"@{CP_LABELS[cp]}": cp for cp in CHECKPOINTS}[cp_glob_m]
        fig = render_importance_bar(
            all_models[cp_key_gm]['importances'], all_models[cp_key_gm]['features'],
            f"Masques d'attention TabNet (moyennés) - @{CP_LABELS[cp_key_gm]}"
        )
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption(f"Barres horizontales - importance moyenne des variables pour TabNet @{CP_LABELS[cp_key_gm]} sur l'ensemble du jeu de test. Une valeur plus élevée indique une variable plus déterminante.")

# ─── Performances de référence ────────────────────────────────────────────────
if match_loaded or manual_stats is not None:
    st.divider()
    st.subheader("Performances de référence")
    perf = pd.DataFrame({
        'Checkpoint':      ['@10 min', '@15 min', '@20 min'],
        'XGBoost Acc':     ['67.96%',  '75.05%',  '78.33%'],
        'TabNet Acc':      ['69.61%',  '76.29%',  '79.27%'],
        'FT-Transf. Acc':  ['69.61%',  '76.24%',  '79.46%'],
        'XGBoost AUC':     ['0.754',   '0.832',   '0.870'],
        'TabNet AUC':      ['0.766',   '0.834',   '0.872'],
        'FT-Transf. AUC':  ['0.766',   '0.839',   '0.876'],
    }).set_index('Checkpoint')
    st.dataframe(perf, use_container_width=True)
    st.caption("9 236 matchs pro · Oracle's Elixir 2025 · FT-Transformer meilleur AUC à partir de @15 min.")
