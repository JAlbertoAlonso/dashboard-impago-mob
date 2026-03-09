# app.py
# Demo Streamlit v0 – Matriz de Impago On-Demand (Datta)
# - Filtros dinámicos N (seleccionas columnas, luego valores)
# - Curva agregada por MOB
# - Matriz display grande
# - Curvas desagregadas (opcional) usando breakdown_col
# - Heatmap básico (opcional, con annot cosechas)
# - Descargas CSV

import io
import re

from typing import Dict, List, Optional, Literal, Tuple

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as path_effects
import plotly.graph_objects as go
import plotly.express as px

from utils.utils_impago_ondemand import ImpagoOnDemandEngine, Scenario
from data.data import load_data

ATRIA_PURPLE = "#783DBE"


# -----------------------------
# Helpers de plots
# -----------------------------

def fig_to_png_bytes(fig: plt.Figure, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def plot_curve_agg(
    curve_df: pd.DataFrame,
    metric_mode_: str,
    tipo_mora: str,
    mob_max: int | None = None,
    title: str = "",
    external_curves: list[tuple[pd.DataFrame, str]] | None = None,
    show_point_labels: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7))

    y_col = "pct_impago_mob" if metric_mode_ == "cosechas" else "pct_ever_mob"

    base = curve_df.copy()
    mob_num = pd.to_numeric(base["MOB"], errors="coerce")
    y_num = pd.to_numeric(base[y_col], errors="coerce")
    mask = mob_num.notna() & y_num.notna()

    if mob_max is not None:
        mask = mask & (mob_num <= int(mob_max))
        mob_max_eff = int(mob_max)
    else:
        mob_max_eff = int(mob_num.loc[mask].max()) if mask.any() else 1

    # Serie completa 1..mob_max_eff para que salgan TODOS los MOBs en eje X
    s = pd.Series(index=range(1, mob_max_eff + 1), dtype=float)
    for m, v in zip(mob_num.loc[mask].astype(int).values, y_num.loc[mask].astype(float).values):
        if 1 <= m <= mob_max_eff:
            s.loc[m] = v

    y = s.values
    y_smooth = smooth_series(y, method="ewm", span=5)

    x = list(range(1, mob_max_eff + 1))

    main_line, = ax.plot(
        x,
        y_smooth,
        ATRIA_PURPLE,
        linewidth=2.2
    )
    main_color = main_line.get_color()

    # Etiquetas por punto (solo si piden)
    if show_point_labels:
        for xi, yi in zip(x, y_smooth):
            if yi is None:
                continue
            try:
                if np.isnan(yi):
                    continue
            except Exception:
                pass

            txt = ax.text(
                xi,
                yi + 0.15,
                f"{float(yi):.1f}",
                fontsize=8,
                ha="center",
                va="bottom",
                color=main_color,
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal()
            ])

    # Curvas externas opcionales
    if external_curves:
        for df_ext, label in external_curves:
            if df_ext is None or df_ext.empty:
                continue

            ext = df_ext.copy()
            ext_m = pd.to_numeric(ext["MOB"], errors="coerce")
            ext_y = pd.to_numeric(ext["value"], errors="coerce")
            ext_mask = ext_m.notna() & ext_y.notna()

            if mob_max is not None:
                ext_mask = ext_mask & (ext_m <= mob_max_eff)

            s2 = pd.Series(index=range(1, mob_max_eff + 1), dtype=float)
            for m, v in zip(ext_m.loc[ext_mask].astype(int).values, ext_y.loc[ext_mask].astype(float).values):
                if 1 <= m <= mob_max_eff:
                    s2.loc[m] = v

            y2 = s2.values
            y2_smooth = smooth_series(y2, method="ewm", span=5)

            ext_line, = ax.plot(
                x,
                y2_smooth,
                linewidth=1.8,
                linestyle="--",
                label=str(label)
            )
            ext_color = ext_line.get_color()

            if show_point_labels:
                for xi, yi in zip(x, y2_smooth):
                    try:
                        if np.isnan(yi):
                            continue
                    except Exception:
                        pass

                    txt = ax.text(
                        xi,
                        yi + 0.15,
                        f"{float(yi):.1f}",
                        fontsize=8,
                        ha="center",
                        va="bottom",
                        color=ext_color,
                    )
                    txt.set_path_effects([
                        path_effects.Stroke(linewidth=3, foreground="white"),
                        path_effects.Normal()
                    ])

    ax.set_xlabel("MOB")
    ax.set_ylabel(build_y_label(tipo_mora, metric_mode_))

    # ticks 1..mob_max_eff
    ax.set_xlim(0, mob_max_eff + 1)
    ax.set_xticks(range(1, mob_max_eff + 1))

    # solo grid horizontal tipo excel
    ax.grid(False)
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=1.0)

    if title:
        ax.set_title(title)

    if external_curves:
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    return fig

def plot_curve_agg_plotly(
    curve_df: pd.DataFrame,
    metric_mode_: str,     
    tipo_mora: str,
    mob_max: int | None = None,
    title: str = "",
    external_curves: list[tuple[pd.DataFrame, str]] | None = None,  # [(df(MOB,value), label), ...]
) -> go.Figure:
    y_col = "pct_impago_mob" if metric_mode_ == "cosechas" else "pct_ever_mob"

    base = curve_df.copy()
    base["MOB_num"] = pd.to_numeric(base["MOB"], errors="coerce")
    base = base.loc[base["MOB_num"].notna()].copy()
    base["MOB_num"] = base["MOB_num"].astype(int)

    if mob_max is not None:
        base = base.loc[base["MOB_num"] <= int(mob_max)].copy()
        mob_max_eff = int(mob_max)
    else:
        mob_max_eff = int(base["MOB_num"].max()) if not base.empty else 1

    base = base.set_index("MOB_num")[[y_col]].sort_index()
    base = base.reindex(range(1, mob_max_eff + 1))

    y = base[y_col].astype(float).values
    y_smooth = smooth_series(y, method="ewm", span=5)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, mob_max_eff + 1)),
        y=y_smooth,
        mode="lines",
        name="Escenario",
        line=dict(
            color=ATRIA_PURPLE,
            width=3
        ),
        hovertemplate="MOB=%{x}<br>%{y:.1f}<extra></extra>",
    ))

    # Curvas externas
    if external_curves:
        for df_ext, label in external_curves:
            if df_ext is None or df_ext.empty:
                continue

            ext = df_ext.copy()
            ext["MOB_num"] = pd.to_numeric(ext["MOB"], errors="coerce")
            ext["value"] = pd.to_numeric(ext["value"], errors="coerce")
            ext = ext.dropna().copy()
            ext["MOB_num"] = ext["MOB_num"].astype(int)
            ext = ext[(ext["MOB_num"] >= 1) & (ext["MOB_num"] <= mob_max_eff)].copy()

            ext = ext.set_index("MOB_num")[["value"]].sort_index()
            ext = ext.reindex(range(1, mob_max_eff + 1))

            y_ext = ext["value"].astype(float).values
            y_ext_smooth = smooth_series(y_ext, method="ewm", span=5)

            fig.add_trace(go.Scatter(
                x=list(range(1, mob_max_eff + 1)),
                y=y_ext_smooth,
                mode="lines",
                name=str(label),
                line=dict(dash="dash"),
                hovertemplate="MOB=%{x}<br>%{y:.1f}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.01
        ),
        height=700,
        xaxis_title="MOB",
        yaxis_title=build_y_label(tipo_mora, metric_mode_),
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),  
        yaxis=dict(showgrid=True, gridcolor="#D9D9D9"),
        xaxis_showgrid=False,
        margin=dict(t=80),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0),
    )
    return fig

def plot_breakdown_curves_plotly(
    curves: list[dict],
    *,
    mob_col: str,
    y_col: str,
    mob_max: int | None,
    title: str,
    tipo_mora: str,
    metric_mode_: str,
) -> go.Figure:
    fig = go.Figure()

    # mob_max efectivo (si no viene, intenta inferir)
    if mob_max is None:
        mob_max_eff = 1
        for it in curves:
            c = it["curve"]
            m = pd.to_numeric(c[mob_col], errors="coerce").dropna()
            if not m.empty:
                mob_max_eff = max(mob_max_eff, int(m.max()))
    else:
        mob_max_eff = int(mob_max)

    # Plot
    for it in curves:
        c = it["curve"].copy()
        c[mob_col] = pd.to_numeric(c[mob_col], errors="coerce")
        c[y_col] = pd.to_numeric(c[y_col], errors="coerce")
        c = c.dropna(subset=[mob_col]).copy()
        c[mob_col] = c[mob_col].astype(int)
        if mob_max is not None:
            c = c[c[mob_col] <= mob_max_eff].copy()

        # reindex a 1..mob_max_eff para que x sea completo
        s = c.set_index(mob_col)[y_col].sort_index()
        s = s.reindex(range(1, mob_max_eff + 1))

        y = s.astype(float).values
        y_smooth = smooth_series(y, method="ewm", span=5)

        fig.add_trace(go.Scatter(
            x=list(range(1, mob_max_eff + 1)),
            y=y_smooth,
            mode="lines",
            name=str(it["label"]),
            hovertemplate="MOB=%{x}<br>%{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
            title=dict(
            text=title,
            x=0.01
        ),
        height=700,
        xaxis_title="MOB",
        yaxis_title=build_y_label(tipo_mora, metric_mode_),
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(showgrid=True, gridcolor="#D9D9D9"),
        xaxis_showgrid=False,
        margin=dict(t=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0
        ),
    )
    return fig

def plot_heatmap_basic(matrix_dt: pd.DataFrame, title: str = "") -> plt.Figure:
    m = matrix_dt.copy()

    # Ordena columnas numéricas si aplica
    try:
        m = m.reindex(sorted(m.columns), axis=1)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(m.values, aspect="auto", cmap="Purples")
    fig.colorbar(im, ax=ax, label="%")

    ax.set_title(title)
    ax.set_xlabel("MOB")
    ax.set_ylabel("Cosecha")

    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels(m.columns)

    ax.set_yticks(range(len(m.index)))
    # Formato cosecha: si es datetime
    ylabels = []
    for idx in m.index:
        try:
            ylabels.append(pd.to_datetime(idx).strftime("%y-%m"))
        except Exception:
            ylabels.append(str(idx))
    ax.set_yticklabels(ylabels)

    # Annot “último valor por fila” con semáforo por percentiles (cosechas)
    arr = m.values.astype(float)
    vals = arr[~np.isnan(arr)]
    if vals.size > 0:
        p50, p80, p95 = np.percentile(vals, [50, 80, 95])

        def semaforo_color(x: float) -> str:
            if x >= p95:
                return "red"
            if x >= p80:
                return "orange"
            if x >= p50:
                return "green"
            return "#1b5e20"

        for i in range(arr.shape[0]):
            row = arr[i, :]
            valid = np.where(~np.isnan(row))[0]
            if len(valid) == 0:
                continue
            j = valid[-1]
            val = row[j]
            ax.text(
                j + 1.5, i, f"{val:.2f}",
                ha="center", va="center",
                color=semaforo_color(val),
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()
    return fig

def format_mob_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    new_cols = []
    for c in out.columns:
        # deja $/# tal cual
        if c in ("$", "#"):
            new_cols.append(c)
            continue

        # columnas MOB: vienen como int o str numérico
        cs = str(c)
        if cs.isdigit():
            new_cols.append(f"MOB<br>{cs}")
        else:
            # cualquier otra columna inesperada, déjala igual
            new_cols.append(cs)

    out.columns = new_cols
    # cosecha: normalmente viene como índice; asegúrate que el índice no tenga nombre
    out.index.name = None
    return out

def _mora_label(tipo_mora_: str) -> str:
    # BGI2+ -> 2+, BGI3+ -> 3+, ... CAST -> CAST
    if isinstance(tipo_mora_, str) and tipo_mora_.startswith("BGI") and tipo_mora_.endswith("+"):
        return tipo_mora_.replace("BGI", "")
    if tipo_mora_ == "CAST":
        return "CAST"
    return str(tipo_mora_)

def _filters_label(filters_: dict[str, list[str]]) -> str:
    
    if not filters_:
        return ""
    parts = []
    for col, val in filters_.items():
        if not val:
            continue
        # compacto: Perfil VIP|STD
        vv = "-".join([str(x) for x in val[:3]])
        if len(val) > 3:
            vv += "-..."
        parts.append(f"{col} {vv}")
    return ", ".join(parts)

def build_plot_title(
    *,
    tipo_mora_: str,
    metric_mode_: str | None = None,
    filters_: dict[str, list[str]],
    castigo_enabled_: bool,
    mob_max_: int,
    breakdown_col_: str | None = None,
    include_detail_: bool = False,
) -> str:
    mora_txt = _mora_label(tipo_mora_)
    fraude_txt = "con fraudes" if castigo_enabled_ else "sin fraudes"
    filtros_txt = _filters_label(filters_)

    chunks = [
        f"Cosechas {mora_txt}@{mob_max_}",
        fraude_txt,
    ]

    if include_detail_ and breakdown_col_:
        chunks.append(f"Detalle {breakdown_col_}")

    if filtros_txt:
        chunks.append(filtros_txt)

    return ", ".join(chunks)

def slug_filename(s: str) -> str:
    # para nombres de archivo (sin comas/espacios raros)
    out = s.strip().lower()
    for ch in [",", ":", ";"]:
        out = out.replace(ch, "")
    out = out.replace(" ", "_")
    out = out.replace("__", "_")
    return out

def slice_matrix_display(df_disp, mob_max):
    # Recorta matriz display manteniendo $/# + MOBs
    cols = list(df_disp.columns)
    extra_cols = [c for c in cols if c in ("$", "#")]
    mob_cols = [c for c in cols if str(c).isdigit() and int(c) <= mob_max]
    return df_disp[extra_cols + mob_cols]

def slice_matrix_dt(df_dt, mob_max):
    # Recorta matrix_dt (numérica) para heatmap
    mob_cols = [c for c in df_dt.columns if int(c) <= mob_max]
    return df_dt[mob_cols]

def smooth_series(y, method: str = "ewm", span: int = 5):
    """
    Suaviza una serie 1D para fines de visualización.
    - method='ewm': media móvil exponencial (recomendado)
    - method='ma' : media móvil simple

    span controla el nivel de suavizado (5-7 suele verse bien).
    """
    s = pd.Series(y).astype(float)
    if method == "ma":
        return s.rolling(window=span, min_periods=1).mean().values
    # default ewm
    return s.ewm(span=span, adjust=False).mean().values

def build_y_label(tipo_mora: str, metric_mode: str) -> str:
    """
    Construye etiqueta eje Y tipo:
    %2+, %3+, %4+, %5+ o %Cast
    """
    if tipo_mora.startswith("BGI"):
        nivel = tipo_mora.replace("BGI", "")
        return f"%{nivel}"
    if tipo_mora == "CAST":
        return "%Cast"
    
    # fallback
    return "% Impago" if metric_mode == "cosechas" else "% Ever"

def normalize_external_curve(ext_input, mob_max: int) -> Optional[pd.DataFrame]:
    """
    Normaliza la curva externa a df con columnas: MOB(int), value(float),
    recortado a mob_max y ordenado. Acepta:
      - DataFrame (Editor / Upload)
      - str (TextArea pegado)
    """
    if ext_input is None:
        return None

    # ---- Caso DataFrame ----
    if isinstance(ext_input, pd.DataFrame):
        df = ext_input.copy()
    else:
        # ---- Caso texto pegado ----
        text = str(ext_input).strip()
        if not text:
            return None

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None

        # Detecta header tipo "MOB,value"
        first = lines[0].lower().replace(" ", "")
        has_header = ("mob" in first) and ("value" in first or "pct" in first)

        data_lines = lines[1:] if has_header else lines

        rows = []
        for ln in data_lines:
            parts = [p for p in re.split(r"[,\t ]+", ln) if p]
            if len(parts) < 2:
                continue
            rows.append((parts[0], parts[1]))

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["MOB", "value"])

    # ---- Normalización columnas ----
    if "MOB" not in df.columns:
        return None

    # columna de valor
    if "value" in df.columns:
        val_col = "value"
    elif "pct" in df.columns:
        val_col = "pct"
    elif "y" in df.columns:
        val_col = "y"
    else:
        # fallback: segunda columna si existe
        if df.shape[1] >= 2:
            val_col = df.columns[1]
        else:
            return None

    mob = pd.to_numeric(df["MOB"], errors="coerce")
    val = pd.to_numeric(df[val_col], errors="coerce")

    out = pd.DataFrame({"MOB": mob, "value": val}).dropna()
    if out.empty:
        return None

    out["MOB"] = out["MOB"].astype(int)
    out = out[(out["MOB"] >= 1) & (out["MOB"] <= int(mob_max))].copy()
    out = out.sort_values("MOB")

    # Si vienen MOB repetidos, promedia
    out = out.groupby("MOB", as_index=False)["value"].mean()

    return out if not out.empty else None

def normalize_external_curve_excel(uploaded_file, mob_max: int) -> pd.DataFrame:
    """
    Lee Excel y valida formato: columnas MOB, value.
    Devuelve df normalizado con MOB(int), value(float), recortado a mob_max.
    """
    df_up = pd.read_excel(uploaded_file)

    # Normaliza nombres por si vienen con espacios/case raro
    df_up = df_up.rename(columns={c: str(c).strip() for c in df_up.columns})

    if "MOB" not in df_up.columns or "value" not in df_up.columns:
        raise ValueError("El Excel debe traer columnas exactamente: MOB, value")

    df = df_up[["MOB", "value"]].copy()
    df["MOB"] = pd.to_numeric(df["MOB"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().copy()
    df["MOB"] = df["MOB"].astype(int)

    df = df[(df["MOB"] >= 1) & (df["MOB"] <= int(mob_max))].copy()
    df = df.sort_values("MOB")
    df = df.groupby("MOB", as_index=False)["value"].mean()  # por si hay repetidos

    return df

def wide_row_template(mob_max: int) -> pd.DataFrame:
    cols = [f"MOB_{i}" for i in range(1, mob_max + 1)]
    return pd.DataFrame([{c: None for c in cols}])

def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    # df_wide: 1 fila con columnas MOB_1..MOB_n
    row = df_wide.iloc[0].to_dict()
    out = []
    for k, v in row.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        # "MOB_12" -> 12
        mob = int(k.split("_")[1])
        out.append((mob, float(v)))
    return pd.DataFrame(out, columns=["MOB", "value"]).sort_values("MOB")

def add_point_labels(ax, x, y, fmt="{:.1f}", dy=0.15):
    """
    Pone etiqueta encima de cada punto (x_i, y_i).
    dy está en unidades del eje Y (ajusta si se ve muy pegado).
    """
    for xi, yi in zip(x, y):
        if yi is None:
            continue
        try:
            if np.isnan(yi):
                continue
        except Exception:
            pass
        ax.text(
            xi, yi + dy,
            fmt.format(float(yi)),
            fontsize=8,
            ha="center",
            va="bottom",
        )

def plot_stacked_composition(agg, bucket_order, breakdown_col, title):
    plot_df = agg.copy()
    plot_df[breakdown_col] = plot_df[breakdown_col].astype(str)

    levels = list(pd.Index(plot_df[breakdown_col].dropna().unique()))
    colors = atria_purple_scale(len(levels))
    color_map = {lvl: colors[i] for i, lvl in enumerate(levels)}

    fig = go.Figure()

    for lvl in levels:
        sub = plot_df[plot_df[breakdown_col] == lvl]
        fig.add_bar(
            x=sub["bucket"],
            y=sub["pct"],
            name=lvl,
            marker_color=color_map[lvl],
        )

    fig.update_layout(
        title=title,
        barmode="stack",
        height=650,
        yaxis_tickformat=".0%",
        xaxis_title="Cosecha",
        yaxis_title="Participación",
        legend_title=breakdown_col,
    )

    fig.update_xaxes(categoryorder="array", categoryarray=bucket_order)

    return fig

def plot_stacked_matplotlib(agg, bucket_order, breakdown_col, title):
    plot_df = agg.copy()
    plot_df[breakdown_col] = plot_df[breakdown_col].astype(str)

    pivot = plot_df.pivot_table(
        index="bucket",
        columns=breakdown_col,
        values="pct",
        fill_value=0
    )

    pivot = pivot.reindex(bucket_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(pivot))

    colors = atria_purple_scale(len(pivot.columns))

    for i, col in enumerate(pivot.columns):
        values = pivot[col].values

        bars = ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            color=colors[i],
            label=col
        )

        for bar, v, b in zip(bars, values, bottom):
            if v > 0.03:
                x = bar.get_x() + bar.get_width() / 2
                y = b + v / 2

                txt = ax.text(
                    x,
                    y,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    clip_on=True
                )
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground="white"),
                    path_effects.Normal()
                ])

        bottom += values

    ax.set_title(title)
    ax.set_ylabel("Participación")
    ax.set_xlabel("Cosecha")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def blend_with_white(hex_color: str, t: float) -> str:
    """
    t=0   -> color original
    t=1.0 -> blanco
    """
    r, g, b = hex_to_rgb(hex_color)
    rr = round(r + (255 - r) * t)
    gg = round(g + (255 - g) * t)
    bb = round(b + (255 - b) * t)
    return rgb_to_hex((rr, gg, bb))

def atria_purple_scale(n: int) -> list[str]:
    """
    Genera n tonos basados en el púrpura Atria.
    De más claro a más oscuro.
    """
    if n <= 1:
        return [ATRIA_PURPLE]

    # claros -> oscuros, evitando casi blanco
    ts = np.linspace(0.68, 0.00, n)
    return [blend_with_white(ATRIA_PURPLE, float(t)) for t in ts]

def plot_transversal_trends_plotly(
    df_plot,
    bucket_order,
    breakdown_col,
    title,
    tipo_mora,
    metric_mode_,
):
    plot_df = df_plot.copy()
    plot_df[breakdown_col] = plot_df[breakdown_col].astype(str)

    levels = list(pd.Index(plot_df[breakdown_col].dropna().unique()))
    levels_wo_total = [lvl for lvl in levels if lvl != "Total"]
    levels_final = levels_wo_total + (["Total"] if "Total" in levels else [])

    # colores default de Plotly
    default_colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ]
    color_map = {
        lvl: default_colors[i % len(default_colors)]
        for i, lvl in enumerate(levels_wo_total)
    }

    # Total con color gris oscuro
    if "Total" in levels_final:
        color_map["Total"] = "#555555"

    fig = go.Figure()

    for lvl in levels_final:
        sub = plot_df[plot_df[breakdown_col] == lvl].copy()
        sub["bucket"] = pd.Categorical(sub["bucket"], categories=bucket_order, ordered=True)
        sub = sub.sort_values("bucket")

        fig.add_trace(
            go.Scatter(
                x=sub["bucket"],
                y=sub["pct"],
                mode="lines",
                name=lvl,
                line=dict(
                    width=3 if lvl == "Total" else 2,
                    color=color_map[lvl],
                    dash="dash" if lvl == "Total" else "solid",
                ),
                hovertemplate=(
                    f"{breakdown_col}: %{{fullData.name}}<br>"
                    "Cosecha: %{x}<br>"
                    "Tasa: %{y:.2%}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        height=650,
        xaxis_title="Cosecha",
        yaxis_title=build_y_label(tipo_mora, metric_mode_),
        yaxis_tickformat=".0%",
        legend_title=breakdown_col,
        hovermode="x unified",
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=bucket_order
    )

    return fig

def plot_transversal_trends_matplotlib(
    df_plot,
    bucket_order,
    breakdown_col,
    title,
    tipo_mora,
    metric_mode_,
    show_point_labels: bool = True,
):
    plot_df = df_plot.copy()
    plot_df[breakdown_col] = plot_df[breakdown_col].astype(str)

    pivot = plot_df.pivot_table(
        index="bucket",
        columns=breakdown_col,
        values="pct",
        fill_value=np.nan
    )

    pivot = pivot.reindex(bucket_order)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Colores default tipo Plotly
    default_colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ]

    cols = list(pivot.columns)
    cols_wo_total = [c for c in cols if c != "Total"]

    color_map = {
        col: default_colors[i % len(default_colors)]
        for i, col in enumerate(cols_wo_total)
    }

    if "Total" in cols:
        color_map["Total"] = "#555555"

    x = np.arange(len(pivot.index))
    x_labels = list(pivot.index)

    for col in cols:
        y = pivot[col].values.astype(float)

        ax.plot(
            x,
            y,
            color=color_map[col],
            linewidth=3 if col == "Total" else 2,
            linestyle="--" if col == "Total" else "-",
            label=col,
        )

        if show_point_labels:
            for xi, yi in zip(x, y):
                try:
                    if np.isnan(yi):
                        continue
                except Exception:
                    continue

                txt = ax.text(
                    xi,
                    yi + 0.002,   # pequeño offset vertical
                    f"{float(yi):.1%}",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color=color_map[col],
                )
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground="white"),
                    path_effects.Normal()
                ])

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    ax.set_xlabel("Cosecha")
    ax.set_ylabel(build_y_label(tipo_mora, metric_mode_))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # solo grid horizontal estilo Excel
    ax.grid(False)
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=1.0)

    if title:
        ax.set_title(title)

    ax.legend(
        title=breakdown_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    fig.tight_layout()
    return fig


# -----------------------------
# Panel de Configuración
# -----------------------------
st.set_page_config(page_title="Demo Matriz de Impago", layout="wide")

# -----------------------------
# Cache data + engine
# -----------------------------
@st.cache_data
def get_df() -> pd.DataFrame:
    return load_data()

df = get_df()

@st.cache_resource
def get_engine(df_: pd.DataFrame) -> ImpagoOnDemandEngine:
    return ImpagoOnDemandEngine(df_)

engine = get_engine(df)

# -----------------------------
# Session state defaults
# -----------------------------

if "last_res" not in st.session_state:
    st.session_state["last_res"] = None

if "last_sc" not in st.session_state:
    st.session_state["last_sc"] = None

if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None

if "reset_epoch" not in st.session_state:
    st.session_state["reset_epoch"] = 0

# -----------------------------
# Sidebar – Controles de escenario
# -----------------------------
epoch = st.session_state["reset_epoch"]

# =============================
# Configuración general
# =============================
st.sidebar.header("Escenario / Cosechas")
tipo_mora = st.sidebar.selectbox(
    "Tipo mora",
    ["BGI2+", "BGI3+", "BGI4+", "BGI5+", "CAST"],
    index=0,
    key=f"tipo_mora_ui_{epoch}",
)

metric_mode = st.sidebar.selectbox(
    "Métrica",
    ["Over", "Ever"],
    index=0,
    key=f"metric_mode_ui_{epoch}",
)
# UI -> Engine
metric_mode_engine = "cosechas" if metric_mode == "Over" else "ever"

castigo_enabled = st.sidebar.checkbox(
    "Con fraude",
    value=False,
    key=f"castigo_enabled_ui_{epoch}",
)

# -----------------------------
# Filtros
# -----------------------------
st.sidebar.markdown("#### Filtros")

FILTERABLE_COLS = [
    "Perfil", "Ciudad Atria", "Plazo", "Antiguedad Unidad",
    "Laboratorios", "Notificaciones", "Atria Plus",
    "MMX 6", "MMX 12", "MMX 24",
    "Cluster Mora", "Cluster Experiencia",
    "Score generico de bancos", "Score Buro",
]

available_cols = [c for c in FILTERABLE_COLS if c in df.columns]

selected_filter_cols = st.sidebar.multiselect(
    "Dimensiones a filtrar",
    options=available_cols,
    default=[],
    key=f"selected_filter_cols_ui_{epoch}",
)

filters: dict[str, list[str]] = {}
for col in selected_filter_cols:
    vals = (
        pd.Series(df[col].astype(str).str.strip().unique())
        .dropna()
        .sort_values()
        .tolist()
    )

    sel = st.sidebar.multiselect(
        f"{col}",
        options=vals,
        default=[],
        key=f"filter_vals_{col}_{epoch}",
    )

    if sel:
        filters[col] = sel

# =============================
# Cap MOB (común)
# =============================
mob_max_cap = int(pd.to_numeric(df["MOB"], errors="coerce").max())

key_heat = f"show_heatmap_ui_{epoch}"
key_mob_table = f"mob_max_table_ui_{epoch}"
key_mob_line = f"mob_max_line_ui_{epoch}"
key_break = f"breakdown_col_ui_{epoch}"
key_minf = f"min_folios_breakdown_ui_{epoch}"

show_heatmap = st.sidebar.checkbox(
    "Mostrar heatmap",
    value=True,
    key=key_heat,
)

mob_max_table_ui = st.sidebar.slider(
    "MOB máximo (matriz / heatmap)",
    min_value=1,
    max_value=mob_max_cap,
    value=min(36, mob_max_cap),
    step=1,
    key=key_mob_table,
)

mob_max_line_ui = st.sidebar.slider(
    "MOB máximo (lineplots)",
    min_value=1,
    max_value=mob_max_cap,
    value=min(24, mob_max_cap),
    step=1,
    key=key_mob_line,
)


# =============================
# Separador
# =============================
st.sidebar.divider()


# =============================
# Detalles Gráficos
# =============================
st.sidebar.header("Escenario / Detalle gráfico")

breakdown_col = st.sidebar.selectbox(
    "Dimensión de detalle",
    options=["(ninguno)"] + available_cols,
    index=0,
    key=key_break,
)

breakdown_col = None if breakdown_col == "(ninguno)" else breakdown_col

min_folios_breakdown = st.sidebar.number_input(
    "# Mínimo de folios",
    min_value=1,
    max_value=5000,
    value=50,
    step=10,
    key=key_minf,
)

st.sidebar.markdown("### Gráfico - Barras apiladas")

freq_mode = st.sidebar.radio(
    "Frecuencia de cosecha",
    ["Mensual", "Trimestral"],
    key=f"stack_freq_ui_{epoch}",
)

n_cosechas = st.sidebar.slider(
    "Número de cosechas",
    min_value=3,
    max_value=24,
    value=6,
    step=1,
    key=f"stack_window_ui_{epoch}",
)

value_mode = st.sidebar.radio(
    "Modo de composición",
    ["Monto fondeado", "Conteo de folios"],
    key=f"stack_value_mode_ui_{epoch}",
)

st.sidebar.markdown("### Gráfico - Tendencias transversales")

trans_freq_mode = st.sidebar.radio(
    "Frecuencia de cosecha",
    ["Mensual", "Trimestral"],
    key=f"trans_freq_ui_{epoch}",
)

trans_n_cosechas = st.sidebar.slider(
    "Número de cosechas",
    min_value=3,
    max_value=24,
    value=6,
    step=1,
    key=f"trans_window_ui_{epoch}",
)

trans_mob_options = list(range(6, mob_max_cap + 1, 6))

if len(trans_mob_options) == 0:
    trans_mob_options = [mob_max_cap]

default_trans_mob = 12 if 12 in trans_mob_options else trans_mob_options[-1]

trans_mob_fix = st.sidebar.select_slider(
    "MOB fijo",
    options=trans_mob_options,
    value=default_trans_mob,
    key=f"trans_mob_fix_ui_{epoch}",
)

# Leer valores anteriores 
show_heatmap = bool(st.session_state.get(key_heat, True))
mob_max_table_ui = int(st.session_state.get(key_mob_table, min(36, mob_max_cap)))
mob_max_line_ui = int(st.session_state.get(key_mob_line, min(24, mob_max_cap)))


# =============================
# Botones
# =============================

# Botón Run
run = st.sidebar.button("Run scenario", type="primary")

# Botón de Reset
def request_reset():
    old = st.session_state["reset_epoch"]
    # borra keys del epoch viejo
    for k in list(st.session_state.keys()):
        if k.endswith(f"_{old}") and (
            k.startswith(("tipo_mora_ui_", "metric_mode_ui_", "castigo_enabled_ui_",
                          "selected_filter_cols_ui_", "breakdown_col_ui_",
                          "min_folios_breakdown_ui_", "show_heatmap_ui_",
                          "mob_max_table_ui_", "mob_max_line_ui_", "filter_vals_"))
        ):
            st.session_state.pop(k, None)

    st.session_state["reset_epoch"] = old + 1

    for k in ["last_res", "last_sc", "last_inputs"]:
        st.session_state.pop(k, None)

st.sidebar.button(
    "Reset",
    type="secondary",
    help="Limpia resultados y regresa a valores por defecto.",
    on_click=request_reset,
)


# ---------------------------------------------------------
# Ejecución
# 1) RUN: solo corre el engine y guarda resultados/inputs
# ---------------------------------------------------------
if run:
    sc = Scenario(
        name="streamlit_demo_v1",
        tipo_mora=tipo_mora,
        metric_mode=metric_mode_engine,
        filters=filters,
        breakdown_col=breakdown_col,
    )

    with st.spinner("Ejecutando escenario..."):
        engine.castigo_enabled = bool(castigo_enabled)
        res = engine.run_scenario(sc, show=False, save_outputs=False, debug=False)

        st.session_state["last_res"] = res
        st.session_state["last_sc"] = sc
        st.session_state["last_inputs"] = {
            "tipo_mora": tipo_mora,
            "metric_mode": metric_mode_engine,
            "filters": filters,
            "breakdown_col": breakdown_col,
            "castigo_enabled": bool(getattr(engine, "castigo_enabled", False)),
        }

    st.success("Listo")

# ---------------------------------------------------------
# 2) RENDER: si ya existe resultado previo
# ---------------------------------------------------------
if st.session_state.get("last_res") is not None:
    res = st.session_state["last_res"]
    sc = st.session_state["last_sc"]
    inputs = st.session_state["last_inputs"]

    tipo_mora = inputs["tipo_mora"]
    metric_mode = inputs["metric_mode"]
    filters = inputs["filters"]
    breakdown_col = inputs["breakdown_col"]
    castigo_enabled = inputs["castigo_enabled"]

    # --- MOBs máxmos mostrados (sliders persistentes)
    mob_max_available = int(
        pd.to_numeric(res["matrix_dt"].columns, errors="coerce").max()
    )
    
    mob_max_table_ui = int(st.session_state.get(f"mob_max_table_ui_{epoch}", min(36, mob_max_cap)))
    mob_max_line_ui  = int(st.session_state.get(f"mob_max_line_ui_{epoch}",  min(24, mob_max_cap)))

    mob_max_table = min(mob_max_table_ui, mob_max_available)
    mob_max_line  = min(mob_max_line_ui,  mob_max_available)

    # --- Recortes consistentes
    curve_view = res["curve_mob"].loc[
        pd.to_numeric(res["curve_mob"]["MOB"], errors="coerce") <= mob_max_line
    ].copy()

    matrix_dt_view = slice_matrix_dt(res["matrix_dt"], mob_max_table)
    matrix_display_view = slice_matrix_display(res["matrix_display"], mob_max_table)

    # --- Títulos (usa MOBs max mostrados)
    title_table = build_plot_title(
        tipo_mora_=tipo_mora,
        metric_mode_=metric_mode,
        filters_=filters,
        castigo_enabled_=bool(castigo_enabled),
        mob_max_=mob_max_table,
        breakdown_col_=None,            
        include_detail_=False,
    )

    title_line_cosechas = build_plot_title(
        tipo_mora_=tipo_mora,
        metric_mode_=metric_mode,
        filters_=filters,
        castigo_enabled_=bool(castigo_enabled),
        mob_max_=mob_max_line,
        breakdown_col_=None,            
        include_detail_=False,
    )

    title_line_detalle = build_plot_title(
        tipo_mora_=tipo_mora,
        metric_mode_=metric_mode,
        filters_=filters,
        castigo_enabled_=bool(castigo_enabled),
        mob_max_=mob_max_line,
        breakdown_col_=breakdown_col,   
        include_detail_=True,           
    )

    file_stub_table = slug_filename(title_table)
    file_stub_line  = slug_filename(title_line_cosechas)
    file_stub_det   = slug_filename(title_line_detalle)

    tab_matriz, tab_break = st.tabs(
        ["Cosechas", "Detalle de gráfico"]
    )


    # ---- TAB 1: Matriz + heatmap + lineplot
    with tab_matriz:
        st.subheader(title_table)

        # -------------------------
        # Matriz (Styler + CSS)
        # -------------------------
        matrix_show = format_mob_headers(matrix_display_view)

        styler = (
            matrix_show.style
            .set_table_styles([
                {"selector": "th", "props": [
                    ("text-align", "center"),
                    ("white-space", "pre-line"),
                    ("font-size", "13px"),
                    ("padding", "2px 4px"),
                ]},
                {"selector": "td", "props": [
                    ("text-align", "center"),
                    ("font-size", "13px"),
                    ("padding", "2px 4px"),
                ]},
            ])
        )

        st.markdown("""
        <style>
        #matrix-wrap table td:first-child,
        #matrix-wrap table th:first-child {
            white-space: nowrap;
            width: 1%;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div id="matrix-wrap">{styler.to_html()}</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            "Descargar matriz (CSV)",
            data=matrix_display_view.to_csv(index=True).encode("utf-8"),
            file_name=f"{file_stub_table}_matriz.csv",
            mime="text/csv",
        )

        st.divider()

        # -------------------------
        # Heatmap (opcional)
        # -------------------------
        if show_heatmap:
            fig_hm = plot_heatmap_basic(
                matrix_dt_view,
                title=title_table,
            )
            png_hm = fig_to_png_bytes(fig_hm)

            st.pyplot(fig_hm, clear_figure=True)
            st.download_button(
                "Heatmap (PNG)",
                data=png_hm,
                file_name=f"{file_stub_table}_heatmap.png",
                mime="image/png",
            )

        st.divider()

        # -------------------------
        # Curvas externas (hasta 3) - SOLO EXCEL
        # -------------------------
        st.markdown("### Comparación: curvas externas (Excel)")

        help_fmt = (
            "Sube hasta 3 archivos Excel (.xlsx/.xls).\n\n"
            "Cada archivo debe tener EXACTAMENTE estas columnas (y en ese orden):\n"
            "MOB, value\n\n"
            "Ejemplo:\n"
            "MOB,value\n"
            "1,0.2\n"
            "2,0.3\n"
            "...\n"
            "24,16.1\n\n"
            "Notas:\n"
            "- MOB debe ser entero.\n"
            "- value numérico.\n"
            "- Si hay MOB repetidos, se promedian.\n"
            f"- Se recorta a MOB <= {mob_max_line}."
        )

        compare_enabled = st.checkbox(
            "Agregar curvas externas",
            value=False,
            key=f"compare_enabled_{epoch}",
            help=help_fmt,
        )

        uploaded_excels = st.file_uploader(
            "Subir Excel(s)",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key=f"ext_excels_{epoch}",
            disabled=not compare_enabled,
            help=help_fmt,
        )

        external_curves = []
        if compare_enabled and uploaded_excels:
            if len(uploaded_excels) > 3:
                st.warning("Subiste más de 3 archivos. Solo se usarán los primeros 3.")
                uploaded_excels = uploaded_excels[:3]

            cols = st.columns(len(uploaded_excels))
            for i, up in enumerate(uploaded_excels, start=1):
                with cols[i-1]:
                    default_label = getattr(up, "name", f"Curva externa {i}")
                    label_i = st.text_input(
                        f"Nombre curva {i}",
                        value=str(default_label).replace(".xlsx", "").replace(".xls", ""),
                        key=f"ext_label_{i}_{epoch}",
                        disabled=not compare_enabled,
                    )

                try:
                    ext_norm = normalize_external_curve_excel(up, mob_max=mob_max_line)
                    if ext_norm is not None and not ext_norm.empty:
                        external_curves.append((ext_norm, label_i))
                    else:
                        st.info(f"Archivo {i}: no se detectaron filas válidas (revisa MOB/value).")
                except Exception as e:
                    st.error(f"Archivo {i}: error leyendo/validando Excel: {e}")

        # -------------------------
        # Curva agregada (escenario + externas si aplica)
        # -------------------------
        fig_line = plot_curve_agg(
            curve_view,
            metric_mode,
            tipo_mora=tipo_mora,
            mob_max=mob_max_line,
            title=title_line_cosechas,
            external_curves=external_curves if (compare_enabled and external_curves) else None,
        )
        png_line = fig_to_png_bytes(fig_line)

        render_mode_tab1 = st.radio(
            "Modo de gráfica",
            ["Interactivo", "Con etiquetas %"],
            index=0,
            horizontal=True,
            key=f"render_mode_tab1_{epoch}",
        )

        use_plotly_tab1 = (render_mode_tab1 == "Interactivo")

        if use_plotly_tab1:
            fig_line = plot_curve_agg_plotly(
                curve_view,
                metric_mode,
                tipo_mora=tipo_mora,
                mob_max=mob_max_line,
                title=title_line_cosechas,
                external_curves=external_curves if (compare_enabled and external_curves) else None,
            )
            st.plotly_chart(fig_line, width="stretch")

        else:
            # Matplotlib con etiquetas
            fig_line = plot_curve_agg(
                curve_view,
                metric_mode,
                tipo_mora=tipo_mora,
                mob_max=mob_max_line,
                title=title_line_cosechas,
                external_curves=external_curves if (compare_enabled and external_curves) else None, 
                show_point_labels=True,  
            )
            st.pyplot(fig_line, clear_figure=True)
        
        st.download_button(
            "Curva agregada (PNG)",
            data=png_line,
            file_name=f"{file_stub_line}_curva.png",
            mime="image/png",
        )


    # ---- TAB 2: Detalle de gráfico
    with tab_break:
        if breakdown_col is None:
            st.info("Selecciona una columna en 'Detalle de gráfico' para ver curvas por nivel.")
        else:

            # -------------------------
            # Plot lineplots
            # -------------------------

            render_mode = st.radio(
                "Modo de gráfica",
                ["Interactivo", "Con etiquetas %"],
                index=0,
                horizontal=False,
                key=f"render_mode_tab2_{epoch}",
            )

            # --- Plotly ---
            if render_mode == "Interactivo":

                curves = engine.compute_curves_by_mob_breakdown(
                    scenario=sc,
                    breakdown_col=breakdown_col,
                    min_folios=int(min_folios_breakdown),
                    mob_max=mob_max_line,
                )

                y_col = "pct_impago_mob" if sc.metric_mode == "cosechas" else "pct_ever_mob"

                fig = plot_breakdown_curves_plotly(
                    curves,
                    mob_col=engine.mob_col,
                    y_col=y_col,
                    mob_max=mob_max_line,
                    title=title_line_detalle,
                    tipo_mora=sc.tipo_mora,
                    metric_mode_=sc.metric_mode,
                )

                st.plotly_chart(fig, width="stretch")

                # export HTML interactivo
                file_bytes = fig.to_html().encode()
                filename = f"{file_stub_det}_detalle_graficos.html"
                mime = "text/html"

            # --- Matplotlib con etiquetas ---
            else:

                fig = engine.plot_curve_by_mob_breakdown(
                    scenario=sc,
                    breakdown_col=breakdown_col,
                    show=False,
                    min_folios=int(min_folios_breakdown),
                    mob_max=mob_max_line,
                    show_point_labels=True,
                )

                # título consistente
                for ax in fig.axes:
                    ax.set_title(title_line_detalle)

                st.pyplot(fig, clear_figure=True)

                file_bytes = fig_to_png_bytes(fig)
                filename = f"{file_stub_det}_detalle_graficos.png"
                mime = "image/png"

            st.download_button(
                "Descargar gráfica PNG",
                data=file_bytes,
                file_name=filename,
                mime=mime,
            )

            st.divider()


            # -------------------------
            # Barras Apiladas
            # -------------------------

            stack_render_mode = st.radio(
                "Modo de gráfica",
                ["Interactivo", "Con etiquetas"],
                key=f"stack_render_ui_{epoch}"
            )

            agg, bucket_order = engine.compute_breakdown_composition(
                sc,
                breakdown_col,
                freq_mode,
                value_mode,
                n_cosechas
            )

            # --- Plotly ---
            if stack_render_mode == "Interactivo":

                fig = plot_stacked_composition(
                    agg,
                    bucket_order,
                    breakdown_col,
                    title_line_detalle
                )

                st.plotly_chart(fig, width="stretch")

                file_bytes = fig.to_html().encode()
                filename = f"{file_stub_det}_barras_apiladas.html"
                mime = "text/html"

            # --- Matplotlib ---
            else:

                fig = plot_stacked_matplotlib(
                    agg,
                    bucket_order,
                    breakdown_col,
                    title_line_detalle
                )

                st.pyplot(fig)

                file_bytes = fig_to_png_bytes(fig)
                filename = f"{file_stub_det}_barras_apiladas.png"
                mime = "image/png"

            st.download_button(
                "Descargar gráfica PNG",
                data=file_bytes,
                file_name=filename,
                mime=mime
            )

            st.divider()

            
            # -------------------------
            # Tendencias transversales
            # -------------------------

            trans_render_mode = st.radio(
                "Modo de gráfica",
                ["Interactivo", "Con etiquetas %"],
                index=0,
                horizontal=True,
                key=f"trans_render_mode_ui_{epoch}",
            )

            trans_df, trans_bucket_order = engine.compute_breakdown_transversal_trends(
                scenario=sc,
                breakdown_col=breakdown_col,
                mob_fix=trans_mob_fix,
                freq_mode=trans_freq_mode,
                n_cosechas=trans_n_cosechas,
            )

            trans_title = f"{title_line_detalle}, MOB {trans_mob_fix} fijo"

            # --- Plotly ---
            if trans_render_mode == "Interactivo":

                fig = plot_transversal_trends_plotly(
                    trans_df,
                    trans_bucket_order,
                    breakdown_col,
                    trans_title,
                    tipo_mora=sc.tipo_mora,
                    metric_mode_=sc.metric_mode,
                )

                st.plotly_chart(fig, width='stretch')

                file_bytes = fig.to_html().encode()
                filename = f"{file_stub_det}_tendencias_transversales.html"
                mime = "text/html"


            # --- Matplotlib ---
            else:

                fig = plot_transversal_trends_matplotlib(
                    trans_df,
                    trans_bucket_order,
                    breakdown_col,
                    trans_title,
                    tipo_mora=sc.tipo_mora,
                    metric_mode_=sc.metric_mode,
                    show_point_labels=True,
                )

                st.pyplot(fig)

                file_bytes = fig_to_png_bytes(fig)
                filename = f"{file_stub_det}_tendencias_transversales.png"
                mime = "image/png"

            st.download_button(
                "Descargar gráfica",
                data=file_bytes,
                file_name=filename,
                mime=mime,
                key=f"download_transversal_{epoch}",
            )

else:
    st.info("Configura el escenario en el panel izquierdo y presiona **Run scenario**.")
