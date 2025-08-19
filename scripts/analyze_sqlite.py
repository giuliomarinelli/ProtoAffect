from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt


def list_runs(conn: sqlite3.Connection) -> pd.DataFrame:
    q = "SELECT id, started_at, cfg_json FROM runs ORDER BY id DESC;"
    return pd.read_sql_query(q, conn)


def load_steps(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    q = "SELECT * FROM steps WHERE run_id=? ORDER BY t;"
    df = pd.read_sql_query(q, conn, params=(run_id,))
    return df


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def maybe_rolling(df: pd.DataFrame, cols: List[str], window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    for c in cols:
        if c in df.columns:
            df[c] = df[c].rolling(window, min_periods=1).mean()
    return df


def plot_timeseries(df: pd.DataFrame, cols: List[str], title: str, outpath: Optional[Path]):
    present = [c for c in cols if c in df.columns]
    if not present:
        print(f"[skip] Nessuna delle colonne richieste per '{title}': {cols}")
        return
    plt.figure(figsize=(10, 4))
    for c in present:
        plt.plot(df["t"], df[c], label=c)
    plt.title(title)
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=140)
        plt.close()
    else:
        plt.show()


def plot_phase(df: pd.DataFrame, title: str, outpath: Optional[Path]):
    if not {"s_valence", "s_arousal"}.issubset(df.columns):
        print(f"[skip] Mancano colonne per '{title}'")
        return
    plt.figure(figsize=(5, 5))
    plt.plot(df["s_valence"], df["s_arousal"])
    plt.xlabel("valence")
    plt.ylabel("arousal")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=140)
        plt.close()
    else:
        plt.show()


def plot_actions(df: pd.DataFrame, outpath: Optional[Path]):
    if "action" not in df.columns:
        print("[skip] Nessuna colonna 'action'")
        return
    # stem plot discreto
    plt.figure(figsize=(10, 2.8))
    plt.stem(df["t"], df["action"], use_line_collection=True)
    plt.yticks(sorted(df["action"].dropna().unique()))
    plt.xlabel("t")
    plt.ylabel("action id")
    plt.title("Azioni nel tempo")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=140)
        plt.close()
    else:
        plt.show()


def export_csv(df: pd.DataFrame, out_csv: Path):
    df.to_csv(out_csv, index=False)
    print(f"[ok] Esportato CSV → {out_csv}")


def main():
    ap = argparse.ArgumentParser(
        description="Analizza data/runs.sqlite e produce grafici/CSV di un run."
    )
    ap.add_argument("--db", default="data/runs.sqlite", help="Percorso al DB SQLite")
    ap.add_argument("--run-id", type=int, default=None, help="ID run da analizzare")
    ap.add_argument("--latest", action="store_true", help="Usa l'ultimo run disponibile")
    ap.add_argument("--tmin", type=int, default=None, help="Filtro minimo su t")
    ap.add_argument("--tmax", type=int, default=None, help="Filtro massimo su t")
    ap.add_argument("--rolling", type=int, default=1, help="Finestra media mobile")
    ap.add_argument("--outdir", default=None, help="Cartella in cui salvare i PNG/CSV")
    ap.add_argument("--no-show", action="store_true", help="Non mostrare grafici a schermo")
    ap.add_argument("--csv", action="store_true", help="Esporta anche CSV dei passi")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"[err] DB non trovato: {args.db}")

    conn = sqlite3.connect(args.db)

    runs = list_runs(conn)
    if runs.empty:
        raise SystemExit("[err] Nessun run presente nel DB.")

    run_id = args.run_id
    if run_id is None:
        # se non specificato, prendi l'ultimo se --latest o comunque il primo della lista
        run_id = int(runs.iloc[0]["id"]) if args.latest or args.run_id is None else None

    if run_id is None:
        print(runs)
        raise SystemExit("[info] Specifica --run-id oppure usa --latest per l'ultimo run.")

    print(f"[info] Analizzo run_id={run_id}")

    df = load_steps(conn, run_id)
    if df.empty:
        raise SystemExit(f"[err] Nessuno step per run_id={run_id}")

    # filtro tempo
    if args.tmin is not None:
        df = df[df["t"] >= args.tmin]
    if args.tmax is not None:
        df = df[df["t"] <= args.tmax]

    # smoothing opzionale
    df = maybe_rolling(
        df,
        cols=[
            "b_energy", "b_safety", "b_belong",
            "s_valence", "s_arousal", "s_dominance",
            "loss", "reward",
        ],
        window=args.rolling,
    )

    outdir = ensure_dir(args.outdir) if args.outdir else None

    # grafici
    if outdir:
        needs_png = outdir / f"run_{run_id:03d}_needs.png"
        affect_png = outdir / f"run_{run_id:03d}_affect.png"
        metrics_png = outdir / f"run_{run_id:03d}_metrics.png"
        phase_png = outdir / f"run_{run_id:03d}_phase.png"
        actions_png = outdir / f"run_{run_id:03d}_actions.png"
    else:
        needs_png = affect_png = metrics_png = phase_png = actions_png = None

    plot_timeseries(df, ["b_energy", "b_safety", "b_belong"], "Bisogni (b) nel tempo", needs_png)
    plot_timeseries(df, ["s_valence", "s_arousal", "s_dominance"], "Affect (s) nel tempo", affect_png)
    plot_timeseries(df, ["loss", "reward"], "Loss/Reward", metrics_png)
    plot_phase(df, "Traiettoria nel piano (valence vs arousal)", phase_png)
    plot_actions(df, actions_png)

    if args.csv and outdir:
        export_csv(df, outdir / f"run_{run_id:03d}_steps.csv")

    if not args.no_show and outdir:
        print(f"[ok] Salvati plot in: {outdir.resolve()}")
    conn.close()


if __name__ == "__main__":
    main()
