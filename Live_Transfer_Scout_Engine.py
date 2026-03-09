# ============================================================
#  LIVE FOOTBALL TRANSFER SCOUT & PLAYER SIMILARITY ENGINE
#  Single-Cell Google Colab Script  |  Big 5 European Leagues
#  Data: Realistic synthetic dataset (FBref-schema compatible)
# ============================================================

# ── 0. IMPORTS ───────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

np.random.seed(42)

# ── 1. SYNTHETIC DATA GENERATION (FBref-schema) ──────────────
# Encodes real football logic:
#   • Position archetypes drive stat distributions
#   • Minutes weighted by squad importance
#   • xG/xAG correlate with goal/assist output + noise
#   • PrgC higher for wingers/strikers; PrgP higher for midfielders
# ─────────────────────────────────────────────────────────────
print("⚽ Generating realistic Big 5 player dataset…")

LEAGUES = {
    'Premier League' : ['Arsenal','Chelsea','Liverpool','Man City','Man Utd','Tottenham',
                        'Newcastle','Aston Villa','Brighton','West Ham'],
    'La Liga'        : ['Real Madrid','Barcelona','Atletico Madrid','Sevilla','Real Sociedad',
                        'Villarreal','Athletic Club','Valencia','Betis','Celta Vigo'],
    'Bundesliga'     : ['Bayern Munich','Dortmund','RB Leipzig','Leverkusen','Frankfurt',
                        'Wolfsburg','Freiburg','Union Berlin','Mainz','Hoffenheim'],
    'Serie A'        : ['Juventus','Inter Milan','AC Milan','Napoli','Roma',
                        'Lazio','Atalanta','Fiorentina','Torino','Bologna'],
    'Ligue 1'        : ['PSG','Marseille','Lyon','Monaco','Rennes',
                        'Lens','Lille','Nice','Strasbourg','Montpellier'],
}

# Positional archetypes: per-90 mean stats
ARCHETYPES = {
    'ST' : dict(Gls=0.52, Ast=0.12, xG=0.48, xAG=0.10, PrgC=3.2, PrgP=2.1),
    'LW' : dict(Gls=0.35, Ast=0.28, xG=0.31, xAG=0.25, PrgC=5.1, PrgP=3.8),
    'RW' : dict(Gls=0.33, Ast=0.30, xG=0.30, xAG=0.27, PrgC=5.3, PrgP=3.6),
    'AM' : dict(Gls=0.28, Ast=0.32, xG=0.26, xAG=0.30, PrgC=4.2, PrgP=5.5),
    'CM' : dict(Gls=0.12, Ast=0.22, xG=0.11, xAG=0.20, PrgC=3.0, PrgP=7.2),
    'DM' : dict(Gls=0.05, Ast=0.10, xG=0.05, xAG=0.09, PrgC=2.1, PrgP=5.8),
    'LB' : dict(Gls=0.04, Ast=0.18, xG=0.04, xAG=0.16, PrgC=3.5, PrgP=4.9),
    'RB' : dict(Gls=0.04, Ast=0.19, xG=0.04, xAG=0.17, PrgC=3.6, PrgP=5.0),
    'CB' : dict(Gls=0.06, Ast=0.05, xG=0.05, xAG=0.04, PrgC=1.2, PrgP=3.8),
}
POS_WEIGHT = {'ST':8,'LW':7,'RW':7,'AM':9,'CM':12,'DM':8,'LB':7,'RB':7,'CB':11}

FIRST = ['Luca','Marco','Kai','Mason','Pedri','Gavi','Vinicius','Kylian','Erling','Phil',
         'Federico','Nicolo','Leandro','Alexis','Theo','Ousmane','Kingsley','Jamal',
         'Florian','Toni','Joshua','Serge','Ilkay','Leon','Thomas','Robert',
         'Antoine','Karim','Lucas','Raphael','Eduardo','Nuno','Enzo','Rodri','Bruno',
         'Marcus','Bukayo','Gabriel','Kai','Granit','Ben','Dani','Jordi',
         'William','Aurelien','Aurelien','Jude','Declan','Kevin','Ruben',
         'Aymeric','Virgil','Andy','Trent','Reece','Kieran','Alexis','Yann']
LAST  = ['Rossi','Muller','Mount','Garcia','Perez','Junior','Mbappe','Haaland','Foden',
         'Bellingham','Chiesa','Barella','Trossard','Sanchez','Hernandez','Dembele',
         'Coman','Musiala','Wirtz','Kroos','Kimmich','Gnabry','Gundogan','Goretzka',
         'Lewandowski','Keane','Griezmann','Benzema','Varane','Camavinga',
         'Silva','Rodri','Fernandes','Rashford','Saka','Martinelli','Havertz',
         'White','Chilwell','James','Trippier','Walker','De Bruyne','Dias',
         'Laporte','van Dijk','Robertson','Alexander-Arnold','Rice',
         'Salah','Firmino','Nunez','Thiago','Jones','Fabinho','Henderson','Elliott',
         'Diaz','Jota','Gomez','Matip','Milner','Oxlade','Wijnaldum','Keita']

rows = []
for league, clubs in LEAGUES.items():
    for club in clubs:
        n_players = np.random.randint(19, 24)
        pos_pool  = np.random.choice(
            list(POS_WEIGHT.keys()), size=n_players,
            p=np.array(list(POS_WEIGHT.values())) / sum(POS_WEIGHT.values())
        )
        for pos in pos_pool:
            arc  = ARCHETYPES[pos]
            mins = int(np.clip(np.random.normal(1800, 650), 400, 3200))
            def noise(mu): return max(0, np.random.normal(mu, mu * 0.35))
            rows.append({
                'Player': f"{np.random.choice(FIRST)} {np.random.choice(LAST)}",
                'Squad' : club, 'Comp': league,
                'Age'   : int(np.clip(np.random.normal(25, 4), 17, 37)),
                'Pos'   : pos,  'Min': mins,
                'Gls'   : round(noise(arc['Gls'])  * mins / 90, 1),
                'Ast'   : round(noise(arc['Ast'])  * mins / 90, 1),
                'xG'    : round(noise(arc['xG'])   * mins / 90, 2),
                'xAG'   : round(noise(arc['xAG'])  * mins / 90, 2),
                'PrgC'  : round(noise(arc['PrgC']) * mins / 90, 1),
                'PrgP'  : round(noise(arc['PrgP']) * mins / 90, 1),
            })

df_raw = pd.DataFrame(rows)

# ── Inject named target + 5 "twin" AM players ────────────────
TARGET_PLAYER = 'Jude Bellingham'   # ← change freely

df_raw = pd.concat([df_raw, pd.DataFrame([{
    'Player':'Jude Bellingham','Squad':'Real Madrid','Comp':'La Liga',
    'Age':20,'Pos':'AM','Min':2650,
    'Gls':18,'Ast':9,'xG':14.2,'xAG':7.8,'PrgC':98,'PrgP':142,
}])], ignore_index=True)

twins = [
    ('Florian Wirtz',  'Leverkusen',   'Bundesliga', 21, 'AM', 2400, 16,11,12.8, 9.1, 88, 158),
    ('Pedri Garcia',   'Barcelona',    'La Liga',    21, 'AM', 2350, 11,13, 9.5,11.2, 76, 171),
    ('Jamal Musiala',  'Bayern Munich','Bundesliga', 21, 'AM', 2500, 15,10,13.1, 8.4,102, 138),
    ('Enzo Fernandez', 'Chelsea',      'Premier League',22,'AM',2200, 8,12, 7.2,10.5, 65, 185),
    ('Gavi Paez',      'Barcelona',    'La Liga',    19, 'AM', 2100, 7,14, 6.8,12.1, 58, 192),
]
for t in twins:
    df_raw = pd.concat([df_raw, pd.DataFrame([{
        'Player':t[0],'Squad':t[1],'Comp':t[2],'Age':t[3],'Pos':t[4],'Min':t[5],
        'Gls':t[6],'Ast':t[7],'xG':t[8],'xAG':t[9],'PrgC':t[10],'PrgP':t[11],
    }])], ignore_index=True)

print(f"   Generated {len(df_raw)} player rows across Big 5 leagues.")

# ── 2. DATA CLEANING & FILTERING ────────────────────────────
print("\n Cleaning data...")

df = df_raw.copy()
for c in ['Age','Min','Gls','Ast','xG','xAG','PrgC','PrgP']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df[~df['Pos'].astype(str).str.contains('GK', na=False)].copy()
df = df[df['Min'] >= 900].copy()
df = df.sort_values('Min', ascending=False).drop_duplicates(subset='Player').reset_index(drop=True)

print(f"   {len(df)} players remaining after GK removal & 900-min filter.")

stat_cols = ['Gls','Ast','xG','xAG','PrgC','PrgP']
p90_cols  = [f'{c}_p90' for c in stat_cols]
for col, p90 in zip(stat_cols, p90_cols):
    df[p90] = (df[col] / df['Min']) * 90
df[p90_cols] = df[p90_cols].fillna(0).replace([np.inf, -np.inf], 0)

print(f"   Per-90 stats computed: {p90_cols}")

# ── 3. ML ENGINE — SIMILARITY & PCA ─────────────────────────
print("\n Running ML engine...")

if TARGET_PLAYER not in df['Player'].values:
    raise ValueError(f"'{TARGET_PLAYER}' not found. Check name spelling.")

X        = df[p90_cols].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca      = PCA(n_components=2, random_state=42)
X_pca    = pca.fit_transform(X_scaled)
df['PCA_1'] = X_pca[:, 0]
df['PCA_2'] = X_pca[:, 1]
print(f"   PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

target_idx = df[df['Player'] == TARGET_PLAYER].index[0]
sim_scores = cosine_similarity(X_scaled[target_idx].reshape(1,-1), X_scaled)[0]
df['Similarity_Score'] = (sim_scores * 100).round(1)

top5 = (df[df['Player'] != TARGET_PLAYER]
        .sort_values('Similarity_Score', ascending=False)
        .head(5).reset_index(drop=True))
top1 = top5.iloc[0]

print(f"   Target   : {TARGET_PLAYER}")
print(f"   #1 Match : {top1['Player']} ({top1['Squad']}) — {top1['Similarity_Score']}%")
print(f"\n   Top-5 Similar Players:")
for i, row in top5.iterrows():
    print(f"   {i+1}. {row['Player']} | {row['Squad']} | {row['Similarity_Score']}%")

# ── 4. PLOTLY VISUALISATIONS ─────────────────────────────────
print("\n Building Plotly figures...")

C_GOLD='#FFD700'; C_GREEN='#39FF14'; C_GRAY='#3a3f5c'
C_BG='#0d1117';   C_PAPER='#161b27'; C_TEXT='#e6edf3'; C_GRID='#21262d'
FONT='IBM Plex Sans, sans-serif'

base_layout = dict(
    paper_bgcolor=C_PAPER, plot_bgcolor=C_BG,
    font=dict(family=FONT, color=C_TEXT, size=12),
    margin=dict(l=40, r=40, t=60, b=40),
)

# FIG 1 — RADAR
radar_metrics = ['xG_p90','xAG_p90','PrgC_p90','PrgP_p90','Ast_p90','Gls_p90']
radar_labels  = ['xG / 90','xAG / 90','Prog Carries','Prog Passes','Assists / 90','Goals / 90']
theta_closed  = radar_labels + [radar_labels[0]]

def radar_vals(name):
    v  = df[df['Player'] == name][radar_metrics].values[0]
    mx = df[radar_metrics].max().values
    mx = np.where(mx == 0, 1, mx)
    return (v / mx) * 10

fig_radar = go.Figure()
for vals, name, colour in [
    (radar_vals(TARGET_PLAYER), TARGET_PLAYER,  C_GOLD),
    (radar_vals(top1['Player']), top1['Player'], C_GREEN),
]:
    r,g,b = int(colour[1:3],16), int(colour[3:5],16), int(colour[5:7],16)
    fig_radar.add_trace(go.Scatterpolar(
        r=list(vals)+[vals[0]], theta=theta_closed, name=name,
        fill='toself', fillcolor=f'rgba({r},{g},{b},0.15)',
        line=dict(color=colour, width=2.5), mode='lines+markers',
        marker=dict(size=6, color=colour),
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.2f}<extra>'+name+'</extra>',
    ))

fig_radar.update_layout(
    **base_layout,
    title=dict(
        text=(f'<b>Player Profile Radar</b><br>'
              f'<span style="font-size:13px;color:#8b949e">'
              f'{TARGET_PLAYER} vs {top1["Player"]}</span>'),
        x=0.5, font=dict(size=18, color=C_TEXT),
    ),
    polar=dict(
        bgcolor=C_BG,
        radialaxis=dict(visible=True, range=[0,10],
                        tickfont=dict(size=9,color='#8b949e'),
                        gridcolor=C_GRID, linecolor=C_GRID),
        angularaxis=dict(tickfont=dict(size=11,color=C_TEXT),
                         gridcolor=C_GRID, linecolor=C_GRID),
    ),
    legend=dict(orientation='h', yanchor='bottom', y=-0.12,
                xanchor='center', x=0.5, font=dict(size=12)),
    height=520,
)

# FIG 2 — PCA SCATTER
top5_names = set(top5['Player'].tolist())
df_rest   = df[(df['Player'] != TARGET_PLAYER) & (~df['Player'].isin(top5_names))]
df_top5   = df[df['Player'].isin(top5_names)]
df_target = df[df['Player'] == TARGET_PLAYER]

fig_scatter = go.Figure()

fig_scatter.add_trace(go.Scatter(
    x=df_rest['PCA_1'], y=df_rest['PCA_2'],
    mode='markers', name='All Players',
    marker=dict(color=C_GRAY, size=5, opacity=0.55, line=dict(width=0)),
    customdata=np.stack([df_rest['Player'],df_rest['Squad'],df_rest['Similarity_Score']],axis=-1),
    hovertemplate=('<b>%{customdata[0]}</b><br>Club: %{customdata[1]}<br>'
                   'Similarity: %{customdata[2]:.1f}%<extra></extra>'),
))

for rank_i, (_, row) in enumerate(df_top5.iterrows(), 1):
    rl = f"#{top5[top5['Player']==row['Player']].index[0]+1} {row['Player']}"
    fig_scatter.add_trace(go.Scatter(
        x=[row['PCA_1']], y=[row['PCA_2']],
        mode='markers+text', name=rl,
        marker=dict(color=C_GREEN, size=13, symbol='diamond',
                    line=dict(color='#ffffff',width=1.2)),
        text=[row['Player'].split()[-1]], textposition='top center',
        textfont=dict(color=C_GREEN, size=10),
        customdata=[[row['Player'],row['Squad'],row['Similarity_Score']]],
        hovertemplate=('<b>%{customdata[0]}</b><br>Club: %{customdata[1]}<br>'
                       'Similarity: %{customdata[2]:.1f}%<extra></extra>'),
        showlegend=(rank_i == 1),
    ))

fig_scatter.add_trace(go.Scatter(
    x=df_target['PCA_1'], y=df_target['PCA_2'],
    mode='markers+text', name=TARGET_PLAYER,
    marker=dict(color=C_GOLD, size=20, symbol='star',
                line=dict(color='#ffffff',width=1.5)),
    text=[TARGET_PLAYER.split()[-1]], textposition='top center',
    textfont=dict(color=C_GOLD, size=12, family=FONT),
    customdata=[[TARGET_PLAYER, df_target['Squad'].values[0], '100.0']],
    hovertemplate=('<b>%{customdata[0]}</b> TARGET<br>Club: %{customdata[1]}<br>'
                   'Similarity: 100%<extra></extra>'),
))

fig_scatter.update_layout(
    **base_layout,
    title=dict(
        text=('<b>PCA Scouting Map</b><br>'
              '<span style="font-size:13px;color:#8b949e">'
              'Big 5 Leagues — positional fingerprint space</span>'),
        x=0.5, font=dict(size=18, color=C_TEXT),
    ),
    xaxis=dict(title='PCA Component 1', gridcolor=C_GRID,
               zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(title='PCA Component 2', gridcolor=C_GRID,
               zeroline=False, tickfont=dict(size=10)),
    legend=dict(bgcolor='rgba(22,27,39,0.85)', bordercolor=C_GRID,
                borderwidth=1, font=dict(size=11)),
    hovermode='closest', height=560,
)

print("   Figures built.")

# ── 5. HTML DASHBOARD ────────────────────────────────────────
print("\n Assembling HTML dashboard...")

radar_html   = fig_radar.to_html(full_html=False, include_plotlyjs='cdn',
                                  config={'displayModeBar': False})
scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False,
                                    config={'displayModeBar': True,
                                            'modeBarButtonsToRemove':['lasso2d','select2d']})

top5_rows_html = ''
for i, row in top5.iterrows():
    bw = row['Similarity_Score']
    top5_rows_html += f"""
        <tr>
          <td class="rank">#{i+1}</td>
          <td class="pname">{row['Player']}</td>
          <td>{row['Squad']}</td>
          <td>{row['Comp']}</td>
          <td>{str(row['Pos']).split(',')[0]}</td>
          <td>
            <div class="bar-wrap">
              <div class="bar-fill" style="width:{bw}%"></div>
              <span class="bar-label">{bw}%</span>
            </div>
          </td>
        </tr>"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Transfer Scout Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root{{--bg:#0d1117;--surface:#161b27;--surface2:#1c2333;--border:#21262d;
  --text:#e6edf3;--muted:#8b949e;--gold:#FFD700;--green:#39FF14;--accent:#388bfd}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'IBM Plex Sans',sans-serif;padding:0 0 60px}}
.header{{background:linear-gradient(135deg,#0d1117 0%,#161b27 60%,#0d2137 100%);
  border-bottom:1px solid var(--border);padding:28px 40px 22px;
  display:flex;align-items:center;gap:18px}}
.header-icon{{font-size:2.4rem}}
.header-titles h1{{font-size:1.55rem;font-weight:700;
  background:linear-gradient(90deg,#FFD700,#FFA500);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.header-titles p{{font-size:0.82rem;color:var(--muted);margin-top:3px;font-family:'IBM Plex Mono',monospace}}
.badge{{margin-left:auto;background:rgba(57,255,20,0.12);border:1px solid var(--green);
  color:var(--green);font-size:0.72rem;font-weight:600;padding:4px 12px;border-radius:20px;letter-spacing:1px}}
.main{{max-width:1380px;margin:0 auto;padding:32px 32px 0}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:32px}}
.kpi-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:20px 24px;position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s}}
.kpi-card:hover{{transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,0,0,0.4)}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0}}
.kpi-card.gold::before{{background:linear-gradient(90deg,#FFD700,#FFA500)}}
.kpi-card.green::before{{background:linear-gradient(90deg,#39FF14,#00cc44)}}
.kpi-card.blue::before{{background:linear-gradient(90deg,#388bfd,#58a6ff)}}
.kpi-card.purple::before{{background:linear-gradient(90deg,#bc8cff,#8957e5)}}
.kpi-label{{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;
  margin-bottom:8px;font-family:'IBM Plex Mono',monospace}}
.kpi-value{{font-size:1.35rem;font-weight:700;line-height:1.2}}
.kpi-sub{{font-size:0.78rem;color:var(--muted);margin-top:4px}}
.kpi-badge{{display:inline-block;font-size:0.7rem;font-weight:600;padding:2px 8px;border-radius:10px;margin-top:6px}}
.badge-gold{{background:rgba(255,215,0,0.15);color:#FFD700}}
.badge-green{{background:rgba(57,255,20,0.12);color:#39FF14}}
.charts-grid{{display:grid;grid-template-columns:1fr 1.4fr;gap:20px;margin-bottom:28px}}
@media(max-width:900px){{.charts-grid{{grid-template-columns:1fr}}}}
.chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden}}
.chart-card-header{{padding:14px 20px 10px;border-bottom:1px solid var(--border);
  font-size:0.78rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;letter-spacing:0.5px}}
.table-card{{background:var(--surface);border:1px solid var(--border);border-radius:14px;
  overflow:hidden;margin-bottom:28px}}
.table-header{{padding:18px 24px;border-bottom:1px solid var(--border);font-size:0.95rem;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:0.86rem}}
thead tr{{background:var(--surface2)}}
th{{padding:10px 16px;text-align:left;font-size:0.72rem;text-transform:uppercase;
  letter-spacing:0.8px;color:var(--muted);font-weight:600;font-family:'IBM Plex Mono',monospace;white-space:nowrap}}
td{{padding:11px 16px;border-bottom:1px solid var(--border)}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(255,255,255,0.025)}}
td.rank{{font-family:'IBM Plex Mono',monospace;color:var(--muted);font-size:0.82rem;width:40px}}
td.pname{{font-weight:600;color:var(--text)}}
.bar-wrap{{position:relative;background:var(--surface2);border-radius:6px;height:20px;width:160px;overflow:hidden}}
.bar-fill{{position:absolute;left:0;top:0;bottom:0;
  background:linear-gradient(90deg,#39FF14,#00cc44);border-radius:6px;opacity:0.8}}
.bar-label{{position:absolute;right:7px;top:50%;transform:translateY(-50%);
  font-size:0.72rem;font-family:'IBM Plex Mono',monospace;color:var(--text);font-weight:600;z-index:1}}
.footer{{text-align:center;padding:20px;font-size:0.75rem;color:var(--muted);
  font-family:'IBM Plex Mono',monospace;border-top:1px solid var(--border);margin-top:16px}}
</style>
</head>
<body>
<div class="header">
  <div class="header-icon">&#9917;</div>
  <div class="header-titles">
    <h1>Transfer Scout Engine</h1>
    <p>Big 5 European Leagues &middot; Player Similarity &amp; PCA Engine &middot; Synthetic FBref-Schema Dataset</p>
  </div>
  <div class="badge">&#9679; SYNTHETIC DATA</div>
</div>
<div class="main">
  <div class="kpi-grid">
    <div class="kpi-card gold">
      <div class="kpi-label">Target Player</div>
      <div class="kpi-value">{TARGET_PLAYER}</div>
      <div class="kpi-sub">{df[df['Player']==TARGET_PLAYER]['Squad'].values[0]} &middot; {df[df['Player']==TARGET_PLAYER]['Pos'].values[0]}</div>
      <span class="kpi-badge badge-gold">REFERENCE</span>
    </div>
    <div class="kpi-card green">
      <div class="kpi-label">#1 Best Match</div>
      <div class="kpi-value">{top1['Player']}</div>
      <div class="kpi-sub">{top1['Squad']} &middot; {str(top1['Pos']).split(',')[0]}</div>
      <span class="kpi-badge badge-green">{top1['Similarity_Score']}% match</span>
    </div>
    <div class="kpi-card blue">
      <div class="kpi-label">Players Analysed</div>
      <div class="kpi-value">{len(df):,}</div>
      <div class="kpi-sub">&ge; 900 min &middot; outfield only</div>
    </div>
    <div class="kpi-card purple">
      <div class="kpi-label">PCA Variance Explained</div>
      <div class="kpi-value">{pca.explained_variance_ratio_.sum()*100:.1f}%</div>
      <div class="kpi-sub">2-component reduction &middot; 6 features</div>
    </div>
  </div>
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-card-header">// RADAR &middot; per-90 profile comparison</div>
      {radar_html}
    </div>
    <div class="chart-card">
      <div class="chart-card-header">// PCA SCOUTING MAP &middot; positional fingerprint space</div>
      {scatter_html}
    </div>
  </div>
  <div class="table-card">
    <div class="table-header">Top 5 Similar Players to <span style="color:var(--gold)">{TARGET_PLAYER}</span></div>
    <table>
      <thead>
        <tr><th>Rank</th><th>Player</th><th>Club</th><th>League</th><th>Position</th><th>Similarity Score</th></tr>
      </thead>
      <tbody>{top5_rows_html}</tbody>
    </table>
  </div>
</div>
<div class="footer">Transfer Scout Engine &middot; Big 5 European Leagues &middot; ML: cosine similarity + PCA &middot; Plotly v5</div>
</body>
</html>"""

output_path = 'Live_Scouting_Engine.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"\n   Dashboard saved to {output_path}")
print("   Triggering download...")
files.download(output_path)
print("\nDone! Open the downloaded HTML file in any browser.")
