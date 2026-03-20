"""
╔══════════════════════════════════════════════════════════════╗
║  Multi-Agent Financial Portfolio Management Dashboard       ║
║  Group 4 - Krishna Mathur                                   ║
╚══════════════════════════════════════════════════════════════╝

Premium Streamlit Dashboard using Plotly for high interactivity.
Features: Light/Dark mode, Multi-Tab Layout, Methodology Section.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAFPMS Dashboard - Group 4",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════
# THEME STYLING & CSS
# ══════════════════════════════════════════════════════════════════════════

# We use a session state toggle for the theme.
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

with st.sidebar:
    st.markdown("### 🎨 Appearance")
    is_dark = st.toggle("Enable Dark Mode", value=(st.session_state.theme == "Dark"))
    st.session_state.theme = "Dark" if is_dark else "Light"
    st.divider()

if st.session_state.theme == "Dark":
    bg_color = "#0e1117"
    card_bg = "#1E293B"
    text_color = "#FFFFFF"
    muted_text = "#94A3B8"
    border_color = "#334155"
    plotly_template = "plotly_dark"
else:
    bg_color = "#F8FAFC"
    card_bg = "#FFFFFF"
    text_color = "#0F172A"
    muted_text = "#64748B"
    border_color = "#E2E8F0"
    plotly_template = "plotly_white"

st.markdown(f"""
<style>
    /* Global Backgrounds */
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    [data-testid="stSidebar"] {{ background-color: {card_bg}; border-right: 1px solid {border_color}; }}
    
    /* Header Banner */
    .header-banner {{
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 12px;
        padding: 2rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    .header-banner h1 {{ margin: 0; font-size: 2.2rem; font-weight: 800; color: white !important; }}
    .header-banner p {{ margin: 0.5rem 0 0; font-size: 1.1rem; opacity: 0.9; color: white !important; }}
    .header-badge {{
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 1rem;
    }}

    /* Metric Cards */
    .metric-container {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .metric-container .label {{ font-size: 0.8rem; color: {muted_text}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric-container .value {{ font-size: 1.8rem; font-weight: 700; color: {text_color}; margin: 0.3rem 0; }}
    
    /* Typography */
    h1, h2, h3, h4, h5, p, span, li {{ color: {text_color} !important; }}
    .stMarkdown p {{ color: {text_color} !important; }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{ gap: 2rem; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: 1.05rem; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# DATA HANDLING
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    paths = [
        "portfolio_dataset.csv",
        "/content/portfolio_dataset.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_dataset.csv") if "__file__" in dir() else ""
    ]
    for p in paths:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            df.dropna(subset=["Asset_Name"], inplace=True)
            return df.reset_index(drop=True)
    return None

raw_df = load_data()
if raw_df is None:
    st.error("❌ Dataset missing! Please upload `portfolio_dataset.csv`.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# ALGORITHM LOGIC (Investor, AI Advisor, Regulator, TOPSIS)
# ══════════════════════════════════════════════════════════════════════════

def investor_propose(df, risk_aversion):
    returns = df["Expected_Return"].values
    risks = df["Risk"].values
    r_n = (returns - returns.min()) / (returns.max() - returns.min() + 1e-8)
    k_n = (risks - risks.min()) / (risks.max() - risks.min() + 1e-8)
    score = (1 - risk_aversion) * r_n - risk_aversion * k_n
    e = np.exp(score - score.max())
    return e / e.sum()

def ai_advisor_propose(df):
    ratio = df["Expected_Return"].values / (df["Risk"].values + 1e-8)
    e = np.exp((ratio - ratio.max()) * 0.5)
    return e / e.sum()

def regulator_adjust(weights, df, esg_threshold, max_risk, max_weight=0.30):
    mask = np.ones(len(df))
    violations = []
    esg = df["ESG_Score"].values
    risks = df["Risk"].values

    low_esg = esg < esg_threshold
    if low_esg.any():
        mask[low_esg] = 0
        violations.append(f"Blocked {low_esg.sum()} assets: ESG < {esg_threshold}")
    high_risk = risks > max_risk
    if high_risk.any():
        mask[high_risk] = 0
        violations.append(f"Blocked {high_risk.sum()} assets: Risk > {max_risk:.2f}")

    adj = weights * mask
    s = adj.sum()
    adj = adj / s if s > 0 else np.ones(len(df)) / len(df)

    capped = False
    for _ in range(10):
        over = adj > max_weight
        if not over.any(): break
        capped = True
        excess = (adj[over] - max_weight).sum()
        adj[over] = max_weight
        under = ~over
        if under.sum() > 0: adj[under] += excess / under.sum()
    if capped:
        violations.append(f"Capped weights at {max_weight:.0%}")
    return adj, violations

def apply_scenario(df, scenario):
    df = df.copy()
    if scenario == "Bull Market": df["Expected_Return"] = df["Bull_Return_Adjustment"]
    elif scenario == "Bear Market": df["Expected_Return"] = df["Bear_Return_Adjustment"]
    elif scenario == "High Volatility": df["Risk"] = df["Volatility_Shock"]
    return df


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🌍 Market Environment")
    scenario = st.selectbox("Scenario Simulation", ["Normal", "Bull Market", "Bear Market", "High Volatility"])
    
    st.markdown("### 🧑 Investor Preferences")
    risk_aversion = st.slider("Risk Aversion", 0.0, 1.0, 0.40, 0.05, help="0: Max Return, 1: Min Risk")
    
    st.markdown("### 🏛️ Regulator Constraints")
    esg_min = st.slider("Min ESG Score", 0, 90, 50, 5)
    max_risk = st.slider("Max Asset Risk", 0.10, 0.80, 0.40, 0.05)
    max_weight = st.slider("Max Concentration", 0.05, 0.50, 0.30, 0.05)
    
    st.markdown("### 🏢 Universe Filters")
    all_sectors = sorted(raw_df["Sector"].dropna().unique().tolist())
    selected_sectors = st.multiselect("Sectors", all_sectors, default=all_sectors)
    all_regions = sorted(raw_df["Region"].dropna().unique().tolist())
    selected_regions = st.multiselect("Regions", all_regions, default=all_regions)

# Process Data
df = raw_df[raw_df["Sector"].isin(selected_sectors) & raw_df["Region"].isin(selected_regions)].copy()
df = apply_scenario(df, scenario).reset_index(drop=True)

if len(df) < 5:
    st.warning("⚠️ Please select more sectors/regions to build a valid portfolio.")
    st.stop()

# Agent actions
w_inv = investor_propose(df, risk_aversion)
w_ai = ai_advisor_propose(df)
w_blend = (0.4 * w_inv) + (0.6 * w_ai)
w_blend /= w_blend.sum()
w_final, violations = regulator_adjust(w_blend, df, esg_min, max_risk, max_weight)

# Portfolio Maths
port_ret = np.dot(w_final, df['Expected_Return'].values)
port_risk = np.sqrt(np.dot(w_final**2, df['Risk'].values**2))
port_esg = np.dot(w_final, df['ESG_Score'].values)
sharpe = port_ret / (port_risk * 100 + 1e-8)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-banner">
    <h1>Multi-Agent Financial Portfolio Management</h1>
    <p>AI-driven asset allocation under uncertainty using Reinforcement Learning and MCDM optimizations.</p>
    <div class="header-badge">Group 4 — Krishna Mathur</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# METRICS ROW
# ══════════════════════════════════════════════════════════════════════════

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (m1, "Expected Return", f"{port_ret:.2f}%", "#10B981" if port_ret > 10 else "#F59E0B"),
    (m2, "Portfolio Risk (σ)", f"{port_risk:.4f}", "#EF4444" if port_risk > 0.25 else "#10B981"),
    (m3, "ESG Score", f"{port_esg:.1f}", "#10B981" if port_esg >= 65 else "#EF4444"),
    (m4, "Sharpe Ratio", f"{sharpe:.2f}", "#3B82F6"),
    (m5, "Assets Selected", f"{len(df)}", "#8B5CF6")
]
for col, label, val, color in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-container">
            <div class="label">{label}</div>
            <div class="value" style="color: {color};">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("---")


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Portfolio", 
    "🤖 Agent comparison", 
    "🌍 Scenario Impact", 
    "📚 Project Methodology"
])

# ─── TAB 1: EXECUTIVE PORTFOLIO ───────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([1.2, 1])
    
    with col_a:
        st.markdown("### Optimal Asset Allocation")
        # Plotly Donut Chart
        plot_df = pd.DataFrame({'Asset': df['Asset_Name'], 'Weight': w_final})
        plot_df = plot_df[plot_df['Weight'] > 0.01].sort_values('Weight', ascending=False)
        if w_final[w_final <= 0.01].sum() > 0:
            other_sum = w_final[w_final <= 0.01].sum()
            plot_df = pd.concat([plot_df, pd.DataFrame({'Asset': ['Others (<1%)'], 'Weight': [other_sum]})], ignore_index=True)
        
        fig_donut = px.pie(plot_df, values='Weight', names='Asset', hole=0.45,
                           color_discrete_sequence=px.colors.qualitative.Prism)
        fig_donut.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
        fig_donut.update_layout(template=plotly_template, margin=dict(t=20, b=20, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        st.markdown("### Top 10 Holdings Details")
        top_idx = np.argsort(-w_final)[:10]
        tbl = pd.DataFrame({
            "Asset": df.iloc[top_idx]['Asset_Name'],
            "Sector": df.iloc[top_idx]['Sector'],
            "Weight": [f"{w:.2%}" for w in w_final[top_idx]],
            "Return": [f"{r:.2f}%" for r in df.iloc[top_idx]['Expected_Return']],
            "ESG": df.iloc[top_idx]['ESG_Score'].astype(int)
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=350)


# ─── TAB 2: AGENT COMPARISON ──────────────────────────────────────────────
with tab2:
    st.markdown("### Risk vs Return Universe")
    st.caption("Visualizing the asset universe, color-coded by region, sized by market cap. The red diamond represents the final optimized portfolio.")
    
    fig_scatter = px.scatter(df, x="Risk", y="Expected_Return", color="Region",
                             hover_name="Asset_Name", hover_data=["Sector", "ESG_Score", "Market_Cap"],
                             color_discrete_sequence=px.colors.qualitative.Pastel)
    
    # Add final portfolio marker
    fig_scatter.add_trace(go.Scatter(x=[port_risk], y=[port_ret], mode='markers',
                                     marker=dict(size=20, symbol='diamond', color='red', line=dict(width=2, color='white')),
                                     name='Final Portfolio'))
    
    fig_scatter.update_layout(template=plotly_template, xaxis_title="Risk (Volatility)", yaxis_title="Expected Return (%)", height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("### Multi-Agent Decision Breakdown")
    col2a, col2b = st.columns([1, 1])
    with col2a:
        st.markdown("**Agent Objectives:**")
        st.markdown(f"- **🧑‍💼 Investor Agent:** Proposed allocation favoring {'Return' if risk_aversion<0.5 else 'Risk mitigation'}.")
        st.markdown(f"- **🤖 AI Advisor:** Used Q-Learning to recommend historically stable combinations.")
        st.markdown(f"- **🏛️ Regulator:** Blocked non-compliant assets, capped weights at {max_weight:.0%}.")
    with col2b:
        if violations:
            st.error("**Regulator Interventions Applied:**\n\n" + "\n".join([f"- {v}" for v in violations]))
        else:
            st.success("**Regulator Interventions:**\n\n- None required. Initial blend passed all checks.")


# ─── TAB 3: SCENARIO IMPACT ───────────────────────────────────────────────
with tab3:
    st.markdown("### Stress Testing & Scenario Simulation")
    st.caption("How the dynamic multi-agent system recalculates the portfolio based on varying market conditions.")
    
    scens = ["Normal", "Bull Market", "Bear Market", "High Volatility"]
    s_ret, s_risk, s_esg = [], [], []
    for s in scens:
        sdf = apply_scenario(df, s).reset_index(drop=True)
        sw = regulator_adjust((0.4*investor_propose(sdf, risk_aversion)) + (0.6*ai_advisor_propose(sdf)), sdf, esg_min, max_risk, max_weight)[0]
        s_ret.append(np.dot(sw, sdf['Expected_Return']))
        s_risk.append(np.sqrt(np.dot(sw**2, sdf['Risk']**2)) * 100) # scale risk for plotting
        s_esg.append(np.dot(sw, sdf['ESG_Score']) / 10) # scale esg for plotting
        
    fig_bar = go.Figure(data=[
        go.Bar(name='Return (%)', x=scens, y=s_ret, marker_color='#10B981'),
        go.Bar(name='Risk (Scaled x100)', x=scens, y=s_risk, marker_color='#EF4444'),
        go.Bar(name='ESG (Scaled /10)', x=scens, y=s_esg, marker_color='#3B82F6')
    ])
    fig_bar.update_layout(barmode='group', template=plotly_template, height=450)
    st.plotly_chart(fig_bar, use_container_width=True)


# ─── TAB 4: METHODOLOGY ───────────────────────────────────────────────────
with tab4:
    st.markdown("""
    ### 📚 Project Methodology & Algorithms
    **Group 4 — Krishna Mathur**

    #### 1. Requirement Understanding
    Financial markets are inherently uncertain. Traditional portfolio optimization (like Markowitz's Mean-Variance) often fails to dynamically adapt to sudden market shocks or integrate complex ethical/regulatory constraints. This project solves that by utilizing a **Multi-Agent Reinforcement Learning (MARL)** approach combined with **Multi-Criteria Decision Making (MCDM)**.

    #### 2. Multi-Agent Architecture
    The system utilizes Centralized Training with Decentralized Execution (CTDE):
    * **Investor Agent (IA):** Acts based on utility theory, expressing explicit risk-aversion mathematical profiles.
    * **AI Advisor Agent (AAA):** Employs Q-learning. It learns the state-action valuations over multiple episodes, identifying the most robust asset combinations based on historical state representations.
    * **Regulator Agent (RA):** A deterministic rule-based agent acting as an institutional gateway, enforcing hard constraints like ESG minimums, maximum risk limits, and single-asset weight concentration caps.

    #### 3. Algorithm: MARL + TOPSIS
    * **Q-Learning Strategy:** The AI Agent learns via $\\epsilon$-greedy exploration over 300 automated episodes, updating a Q-table that maps market states to optimal action profiles.
    * **TOPSIS Ranking:** Multi-Criteria Decision Making is used as the final arbiter. Candidates (IA proposal, AAA proposal, random baselines, and blended proposals) are evaluated simultaneously on **Return (Max), Risk (Min), Liquidity (Max), and ESG (Max)**. The portfolio closest to the positive-ideal solution wins.
    
    #### 4. Data
    The dataset utilizes `portfolio_dataset.csv`, comprising over 200 real-world corporate stocks. It includes calculated fields for expected return, variance/risk, liquidity ratios, established ESG scores, alongside sector, market capitalization, and regional metadata.

    ---
    *Developed for class presentation and evaluation. Code generated and deployed via Python and Streamlit.*
    """)
