"""Entrenamiento SUPERVISADO de pronóstico de condición adversa.

Uso:
    python -m src.train_supervised        # usa Azure SQL (DATA_SOURCE=sql)

Objetivo: P(severidad >= adverse_min en (t, t+H]) desde la ventana de metales.
Verdad de campo = condición del laboratorio unificada (ver src/data/labels.py).

Pipeline:
  1. Carga metales (densos) + condición (dispersa); etiqueta por horizonte con censura.
  2. Escala features, construye ventanas supervisadas (solo desenlace observable).
  3. Split TEMPORAL con embargo (sin fuga): entrena en el pasado, evalúa en el futuro.
  4. Entrena LSTMClassifier con pérdida ponderada por desbalance (pos_weight).
  5. Baseline HistGradientBoosting sobre features de tendencia (rápido y explicable).
  6. Reporta PR-AUC, ROC-AUC, recall@precisión y matriz de confusión en TEST.
  7. Guarda artefactos en artifacts/.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

from .config import ARTIFACTS, load_config
from .data.labels import label_future_adverse, load_condition_series
from .data.load import load_data
from .features.windows import FleetScaler, make_supervised_windows
from .models.classifier import LSTMClassifier


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


_STATS = ["last", "mean", "std", "max", "min", "slope", "delta"]


def artifact_suffix(cfg: dict) -> str:
    """Sufijo de artefactos por componente y severidad: _<comp>_sev{2|3}.
    Permite que coexistan modelos de motor, transmisión, etc. sin pisarse."""
    comp = cfg.get("_component", "engine")
    return f"_{comp}_sev{int(cfg['target']['adverse_min_severity'])}"


def feature_names(feat_cols) -> list[str]:
    """Nombres alineados con window_features (bloques en orden _STATS)."""
    return [f"{s}_{c}" for s in _STATS for c in feat_cols]


def window_features(X: np.ndarray) -> np.ndarray:
    """Resumen tabular por ventana para el baseline GBT: last, mean, std, max, min,
    slope (pendiente temporal) y delta (last-first) por feature."""
    N, T, F = X.shape
    tt = np.arange(T)
    tt_c = tt - tt.mean()
    denom = (tt_c ** 2).sum() or 1.0
    last, first = X[:, -1, :], X[:, 0, :]
    mean = X.mean(1)
    slope = ((X - mean[:, None, :]) * tt_c[None, :, None]).sum(1) / denom
    return np.concatenate([last, mean, X.std(1), X.max(1), X.min(1), slope, last - first], axis=1)


def _temporal_split(dates: np.ndarray, embargo_days: int):
    """Split temporal por fecha del ancla con embargo entre train y val/test."""
    d = dates.astype("datetime64[D]").astype(int)
    cut_tr, cut_te = np.quantile(d, 0.70), np.quantile(d, 0.85)
    train = d < (cut_tr - embargo_days)
    val = (d >= cut_tr) & (d < cut_te)
    test = d >= cut_te
    return train, val, test


def _group_split(grp: np.ndarray, seed: int):
    """Split por equipo (holdout de motores): generaliza a equipos NO vistos."""
    rng = np.random.default_rng(seed)
    eqs = np.array(sorted(set(grp.tolist())))
    rng.shuffle(eqs)
    n = len(eqs)
    tr_eq = set(eqs[: int(0.70 * n)])
    va_eq = set(eqs[int(0.70 * n): int(0.85 * n)])
    train = np.array([e in tr_eq for e in grp])
    val = np.array([e in va_eq for e in grp])
    test = ~(train | val)
    return train, val, test


def _threshold_at_precision(y_true, prob, target):
    """Umbral operativo: máximo F1 ENTRE los umbrales con precisión >= target.
    Usar F1 (no la mínima cobertura) evita caer en picos de ruido a umbral ~0.
    Si la precisión objetivo es inalcanzable, cae al percentil 90 de prob."""
    prec, rec, ths = precision_recall_curve(y_true, prob)
    prec, rec = prec[:-1], rec[:-1]
    ok = prec >= target
    if not ok.any():
        return float(np.quantile(prob, 0.90))
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    f1 = np.where(ok, f1, -1.0)
    return float(ths[int(np.argmax(f1))])


def _report(name: str, y_true: np.ndarray, prob: np.ndarray, thr: float | None = None):
    ap = average_precision_score(y_true, prob)
    try:
        roc = roc_auc_score(y_true, prob)
    except ValueError:
        roc = float("nan")
    print(f"\n[{name}]  PR-AUC={ap:.3f}  ROC-AUC={roc:.3f}  (base rate={y_true.mean():.3f})")
    prec, rec, ths = precision_recall_curve(y_true, prob)
    for p_target in (0.5, 0.7):
        ok = prec[:-1] >= p_target
        if ok.any():
            i = np.argmax(rec[:-1] * ok)
            print(f"   recall @ precision>={p_target:.0%}: {rec[:-1][i]:.2f} "
                  f"(thr={ths[i]:.3f})")
        else:
            print(f"   precision>={p_target:.0%}: inalcanzable")
    if thr is not None:
        yhat = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
        print(f"   @thr={thr:.3f}: TP={tp} FP={fp} FN={fn} TN={tn}  "
              f"prec={tp/max(1,tp+fp):.2f} rec={tp/max(1,tp+fn):.2f}")
    return ap


def main():
    os.environ.setdefault("DATA_SOURCE", "sql")
    cfg = load_config()
    _set_seed(cfg["train"]["seed"])
    ARTIFACTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}  | DATA_SOURCE={os.environ['DATA_SOURCE']}")

    # ---- 1. Datos + etiqueta ----
    df = load_data(cfg)
    cond = load_condition_series(cfg)
    year_min = cfg["target"].get("year_min")
    if year_min:
        df = df[df["fecha_muestra"].dt.year >= int(year_min)].reset_index(drop=True)
        cond = cond[cond["fecha_muestra"].dt.year >= int(year_min)].reset_index(drop=True)
        print(f"Filtro era: solo año >= {year_min}  -> {len(df)} muestras")
    lab = label_future_adverse(df[["equipo", "fecha_muestra"]], cfg, condition_df=cond)
    lab = lab.rename(columns={"equipo": "_eq"}).drop_duplicates(["_eq", "fecha_muestra"])
    df["_eq"] = df["equipo"].astype(str).str.upper()
    df = df.merge(lab[["_eq", "fecha_muestra", "y_target", "label_valido", "lead_time_dias"]],
                  on=["_eq", "fecha_muestra"], how="left")
    df["label_valido"] = df["label_valido"].fillna(False)
    df["y_target"] = df["y_target"].fillna(0).astype(int)

    # ---- 2. Ventanas CRUDAS (el escalado se ajusta DESPUÉS del split, sin fuga) ----
    feat_cols = cfg["oil_vars"] + cfg["context_vars"] + cfg.get("extra_vars", [])
    scaler = FleetScaler(feat_cols)
    X_raw, y, grp, dates = make_supervised_windows(df, cfg, scaler=None)
    print(f"Ventanas etiquetadas: {X_raw.shape} | positivos={int(y.sum())} ({y.mean():.1%})")
    if len(X_raw) == 0 or y.sum() == 0:
        raise SystemExit("Sin ventanas etiquetadas/positivas. Revisa target.* en config.")

    # ---- 3. Split (temporal con embargo | por equipo) ----
    split_mode = cfg["train"].get("split_mode", "temporal")
    if split_mode == "group":
        tr, va, te = _group_split(grp, cfg["train"]["seed"])
    else:
        tr, va, te = _temporal_split(dates, cfg["target"]["horizon_days"])
    print(f"Split [{split_mode}] -> train={tr.sum()} (pos={y[tr].sum()})  "
          f"val={va.sum()} (pos={y[va].sum()})  test={te.sum()} (pos={y[te].sum()})")
    for name, m in [("train", tr), ("val", va), ("test", te)]:
        if m.sum() == 0 or y[m].sum() == 0:
            raise SystemExit(f"Split '{name}' sin positivos; ajusta cortes/horizonte.")

    # Escalado: estadísticos ajustados SOLO en train, aplicados a todo (sin fuga).
    scaler.fit_on_array(X_raw[tr].reshape(-1, X_raw.shape[2]))
    X = scaler.transform_windows(X_raw)

    # ---- 4. LSTM clasificador ----
    pos_weight = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())], dtype=torch.float32)
    print(f"pos_weight={pos_weight.item():.2f}")
    clf = LSTMClassifier(input_dim=X.shape[2], hidden_dim=cfg["model"]["lstm_hidden"],
                         num_layers=cfg["model"]["lstm_layers"],
                         dropout=cfg["model"]["mc_dropout"]).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"].get("weight_decay", 0.0))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    dl = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr], dtype=torch.float32)),
                    batch_size=cfg["train"]["batch_size"], shuffle=True)
    Xva = torch.tensor(X[va]).to(device)

    best_ap, best_state, patience, wait = -1.0, None, 12, 0
    for ep in range(80):
        clf.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(clf(xb), yb).backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(clf(Xva)).cpu().numpy()
        ap = average_precision_score(y[va], p_va)
        if ap > best_ap:
            best_ap, best_state, wait = ap, {k: v.cpu().clone() for k, v in clf.state_dict().items()}, 0
        else:
            wait += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:>3}  val PR-AUC={ap:.3f}  (best={best_ap:.3f})")
        if wait >= patience:
            print(f"  early stop en epoch {ep+1} (best val PR-AUC={best_ap:.3f})")
            break
    clf.load_state_dict(best_state)

    # Umbral por precisión objetivo en validación; MC Dropout para test
    clf.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(clf(Xva)).cpu().numpy()
    pt = cfg["train"].get("precision_target", 0.6)
    pt_alta = cfg["train"].get("precision_target_alta", 0.85)
    # Calibración isotónica (ajustada en val): la prob mostrada es interpretable y
    # los umbrales de banda viven en espacio de probabilidad real, no de score.
    iso_lstm = IsotonicRegression(out_of_bounds="clip").fit(p_va, y[va])
    p_va_c = iso_lstm.transform(p_va)
    thr = _threshold_at_precision(y[va], p_va_c, pt)
    thr_alta = _threshold_at_precision(y[va], p_va_c, pt_alta)
    p_te, p_te_std = clf.predict_proba(torch.tensor(X[te]).to(device),
                                       n_samples=cfg["model"]["mc_samples"])
    p_te = iso_lstm.transform(p_te)
    ap_lstm = _report("LSTM clasificador (TEST)", y[te], p_te, thr)
    print(f"   incertidumbre media (std MC) en test: {p_te_std.mean():.3f}")

    # ---- 5. Baseline gradient boosting ----
    gbt = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_depth=3,
        l2_regularization=1.0, class_weight="balanced",
        random_state=cfg["train"]["seed"])
    gbt.fit(window_features(X[tr]), y[tr])
    p_va_gbt = gbt.predict_proba(window_features(X[va]))[:, 1]
    p_te_gbt = gbt.predict_proba(window_features(X[te]))[:, 1]
    iso_gbt = IsotonicRegression(out_of_bounds="clip").fit(p_va_gbt, y[va])
    p_va_gbt = iso_gbt.transform(p_va_gbt)
    p_te_gbt = iso_gbt.transform(p_te_gbt)
    thr_gbt = _threshold_at_precision(y[va], p_va_gbt, pt)
    thr_gbt_alta = _threshold_at_precision(y[va], p_va_gbt, pt_alta)
    ap_gbt = _report("GradientBoosting tendencia (TEST)", y[te], p_te_gbt, thr_gbt)

    # Explicabilidad: importancia por permutación (qué features mueven el riesgo)
    try:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(gbt, window_features(X[te]), y[te], n_repeats=5,
                                    random_state=cfg["train"]["seed"], scoring="average_precision")
        names = feature_names(feat_cols)
        top = np.argsort(pi.importances_mean)[::-1][:12]
        print("\n[GBT] Top features (importancia por permutación, caída de PR-AUC):")
        for i in top:
            print(f"   {names[i]:<16} {pi.importances_mean[i]:+.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"  (importancia omitida: {e})")

    # ---- 6. Lead time de los positivos detectados (LSTM @thr) ----
    lt = df.loc[df["label_valido"] & (df["y_target"] == 1), "lead_time_dias"].dropna()
    if len(lt):
        print(f"\nLead time (días) en positivos: media={lt.mean():.0f} mediana={lt.median():.0f}")

    # ---- 7. Guardar ----
    mejor = "LSTM" if ap_lstm >= ap_gbt else "GBT"
    print(f"\nMejor por PR-AUC: {mejor}  (LSTM={ap_lstm:.3f} vs GBT={ap_gbt:.3f})")
    sfx = artifact_suffix(cfg)   # _sev2 (screen general) | _sev3 (alerta crítica)
    torch.save(clf.state_dict(), ARTIFACTS / f"clf_lstm{sfx}.pt")
    joblib.dump(gbt, ARTIFACTS / f"clf_gbt{sfx}.joblib")
    joblib.dump(scaler, ARTIFACTS / f"scaler_sup{sfx}.joblib")
    joblib.dump({"thr_lstm": thr, "thr_lstm_alta": thr_alta,
                 "thr_gbt": thr_gbt, "thr_gbt_alta": thr_gbt_alta,
                 "iso_lstm": iso_lstm, "iso_gbt": iso_gbt,
                 "input_dim": X.shape[2], "feat_cols": feat_cols, "mejor": mejor,
                 "ap_lstm": ap_lstm, "ap_gbt": ap_gbt,
                 "component": cfg.get("_component", "engine"),
                 "adverse_min": int(cfg["target"]["adverse_min_severity"])},
                ARTIFACTS / f"meta_sup{sfx}.joblib")
    print(f"Artefactos guardados en {ARTIFACTS}/  (clf_lstm{sfx}.pt, clf_gbt{sfx}.joblib, ...)")
    print("Ejecuta `python -m src.predict_supervised` para el tablero de flota.")


if __name__ == "__main__":
    main()
