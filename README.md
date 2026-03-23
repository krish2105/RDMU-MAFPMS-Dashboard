# 📊 Multi-Agent Financial Portfolio Management under Uncertainty

**Group 4 - Krishna Mathur**
**Module:** Multi-Agent Systems

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🌐 Live Dashboard
**Interact with the live dashboard here:**  
👉 **[https://rdmu-mafpms-dashboard-akreerdaumz4epgywpxeea.streamlit.app/](https://rdmu-mafpms-dashboard-akreerdaumz4epgywpxeea.streamlit.app/)**

---

## 📖 Project Overview
Financial markets operate under severe uncertainty, rendering traditional portfolio optimization fragile. This project architects a robust **Multi-Agent System (MAS)** capable of dynamic asset allocation. 

The system natively understands mathematically defined risk-aversion, optimizes over historical market states using machine learning, and strictly enforces real-world institutional constraints (ESG mandates, concentration limits).

### 🤖 The Multi-Agent Architecture
This system utilizes Centralized Training with Decentralized Execution via three distinct autonomous agents:
1. **🧑‍💼 Investor Agent:** Proposes allocations purely based on a mathematical utility-driven risk-aversion profile (0.0 Aggressive -> 1.0 Conservative).
2. **🧠 AI Advisor Agent:** Uses **Reinforcement Learning (Q-Learning)** with an $\epsilon$-greedy exploration strategy over hundreds of episodes to map optimal market states to historical action profiles.
3. **🏛️ Regulator Agent:** A deterministic gatekeeper enforcing hard constraints (Minimum ESG bounds, Risk ceilings, Max concentration weights).

### 🧮 Algorithm (MARL + TOPSIS)
Once the independent proposals are aggregated and regulated, **Multi-Criteria Decision Making (TOPSIS)** ranks all candidate portfolios concurrently across 4 vectors: 
- Return (+)
- Risk (-)
- Liquidity (+)
- ESG (+)

The portfolio positioned closest to the positive-ideal Euclidean distance is executed as the final output.

---

## 💻 Tech Stack
- Frontend: **Streamlit**, **Plotly**
- Backend: **Python 3.10+** (Pandas, Numpy)
- Algorithms: MARL (Q-Learning proxy), MCDM (TOPSIS)

## 📂 File Structure
- `app.py` — The interactive visualization Streamlit dashboard.
- `portfolio_dataset.csv` — The dataset comprising 200+ global assets with synthetic stress-test vectors.
- `requirements.txt` — Dependency list for cloud deployment.
- `MAFPM_System.py` — The headless Python engine for executing the algorithms directly.
