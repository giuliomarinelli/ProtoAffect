# proto-affect — PoC-1: un nucleo affettivo artificiale

> Obiettivo: costruire un agente **minuscolo ma vivo** che mantiene **bisogni** (omeostasi), uno **stato affettivo** continuo (valence, arousal, dominance), un **appraisal** degli eventi, una **policy stocastica** e **memoria**. Niente etichette “gioia/tristezza”: solo dinamiche interne osservabili con persistenza e isteresi.

---

## TL;DR (Quickstart)

```bash
# 1) Python 3.11 consigliato
python -V

# 2) Ambiente e dipendenze (CPU)
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install torch numpy pandas matplotlib pyyaml

# 3) Avvia training minimale
python scripts/train.py

# 4) Analizza l'ultimo run e salva i plot
python scripts/analyze_sqlite.py --latest --outdir outputs/plots --no-show --csv
