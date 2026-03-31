# app.py
# Demo Streamlit – Matriz de Impago On-Demand (Datta)
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


# =================================
# Helpers de plots
# =================================

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

    if y_col not in curve_df.columns:
        available_cols = list(curve_df.columns)
        raise KeyError(
            f"La curva agregada no contiene la columna esperada '{y_col}'. "
            f"metric_mode_={metric_mode_}. Columnas disponibles: {available_cols}"
        )

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

    y = s.values.astype(float)
    x = list(range(1, mob_max_eff + 1))

    main_line, = ax.plot(
        x,
        y,
        ATRIA_PURPLE,
        linewidth=2.2
    )

    main_color = main_line.get_color()

    # Etiquetas por punto (solo si piden)
    if show_point_labels:
        for xi, yi in zip(x, y):
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

            y2 = s2.values.astype(float)

            ext_line, = ax.plot(
                x,
                y2,
                linewidth=1.8,
                linestyle="--",
                label=str(label)
            )

            ext_color = ext_line.get_color()

            if show_point_labels:
                for xi, yi in zip(x, y2):
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

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, mob_max_eff + 1)),
        y=y,
        mode="lines",
        name="Escenario",
        line=dict(
            color=ATRIA_PURPLE,
            width=3
        ),
        hovertemplate="MOB %{x}<br>%{y:.1f}%<extra></extra>",
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

            fig.add_trace(go.Scatter(
                x=list(range(1, mob_max_eff + 1)),
                y=y_ext,
                mode="lines",
                name=str(label),
                line=dict(dash="dash"),
                hovertemplate="MOB %{x}<br>%{y:.1f}%<extra></extra>",
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
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)",
            bordercolor=ATRIA_PURPLE,
            font=dict(
                size=16,
                color="#1f1f1f"
            )
        )
    )
    fig = apply_axis_style(fig)

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

    # mob_max efectivo
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

        fig.add_trace(go.Scatter(
            x=list(range(1, mob_max_eff + 1)),
            # y=y_smooth,
            y=y,
            mode="lines",
            name=str(it["label"]),
            hovertemplate="MOB %{x}<br>%{y:.1f}%<extra></extra>",
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
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            itemsizing="constant",
            # traceorder="normal",
        ),
        margin=dict(t=80, r=220),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)",
            bordercolor="#783DBE",
            font=dict(
                size=16,
                color="#1f1f1f"
            )
        )
    )

    fig = apply_axis_style(fig)

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
            d = pd.to_datetime(idx, format="%Y-%m", errors="coerce")
            ylabels.append(d.strftime("%y-%m") if pd.notna(d) else str(idx))
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
        vv = " ".join([str(x) for x in val[:5]])
        if len(val) > 5:
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

    if breakdown_col_:
        title_ = f"Cosechas {mora_txt}"
    else:
        title_ = f"Cosechas {mora_txt}@{mob_max_}"

    chunks = [
        title_,
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

def plot_stacked_plotly(agg, bucket_order, breakdown_col, title):
    plot_df = agg.copy()
    plot_df[breakdown_col] = plot_df[breakdown_col].astype(str)

    pivot = plot_df.pivot_table(
        index="bucket",
        columns=breakdown_col,
        values="pct",
        fill_value=0
    )

    pivot = pivot.reindex(bucket_order, fill_value=0)

    # Defensivo si hay problema en los buckets
    row_sums = pivot.sum(axis=1).replace(0, np.nan)
    pivot = pivot.div(row_sums, axis=0).fillna(0)

    levels = sorted(pivot.columns)
    colors = [
        "#783DBE",
        "#5B8FF9",
        "#61DDAA",
        "#65789B",
        "#F6BD16",
        "#7262FD",
        "#78D3F8",
        "#9661BC",
        "#F6903D",
        "#008685",
    ]

    color_map = {lvl: colors[i % len(colors)] for i, lvl in enumerate(levels)}

    fig = go.Figure()

    for lvl in levels:
        fig.add_bar(
            x=pivot.index.tolist(),
            y=pivot[lvl].tolist(),
            name=lvl,
            marker_color=color_map[lvl],
            hovertemplate=(
                f"{breakdown_col}: {lvl}<br>"
                "Cosecha: %{x}<br>"
                "Participación: %{y:.1%}<extra></extra>"
            ),
        )

    layout_kwargs = dict(
        barmode="stack",
        height=650,
        yaxis_tickformat=".0%",
        xaxis_title="Cosecha",
        yaxis_title="Participación",
        legend=dict(
            title=breakdown_col,
            traceorder="normal"
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)",
            bordercolor=ATRIA_PURPLE,
            font=dict(
                size=16,
                color="#1f1f1f"
            )
        ),
        margin=dict(t=20, r=40, l=40, b=40),
    )

    if title:
        layout_kwargs["title"] = dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=20, color="#1f1f1f"),
        )
        layout_kwargs["margin"] = dict(t=70, r=40, l=40, b=40)

    fig.update_layout(**layout_kwargs)

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=bucket_order,
        tickangle=315
    )

    fig = apply_axis_style(fig)

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

    fig, ax = plt.subplots(figsize=(12, 12))
    bottom = np.zeros(len(pivot))

    # colors = atria_purple_scale(len(pivot.columns))
    colors = [
        "#783DBE",
        "#5B8FF9",
        "#61DDAA",
        "#65789B",
        "#F6BD16",
        "#7262FD",
        "#78D3F8",
        "#9661BC",
        "#F6903D",
        "#008685",
    ]

    for i, col in enumerate(pivot.columns):
        values = pivot[col].values

        bars = ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            color=colors[i % len(colors)],
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
                    fontsize=12,
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
    ax.tick_params(axis="x", labelrotation=45)

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
    fixed_y_0_100: bool = False,
):
    plot_df = df_plot.copy()
    plot_df[breakdown_col] = plot_df[breakdown_col].astype(str)

    # -----------------------------
    # Normalización de buckets:
    # - si vienen en formato YYYY-MM -> se normalizan a YYYY-MM
    # - si vienen trimestrales (YYYY_I, YYYY_II, ...) -> se dejan tal cual
    # -----------------------------
    bucket_order_raw = [str(x) for x in bucket_order]

    monthly_pattern = re.compile(r"^\d{4}-\d{2}$")
    is_monthly_bucket = all(monthly_pattern.match(x) for x in bucket_order_raw)

    if is_monthly_bucket:
        parsed_order = pd.to_datetime(bucket_order_raw, format="%Y-%m", errors="coerce")
        bucket_order_fmt = [d.strftime("%Y-%m") for d in parsed_order]

        bucket_map = {
            raw: fmt for raw, fmt in zip(bucket_order_raw, bucket_order_fmt)
        }

        plot_df["bucket"] = (
            plot_df["bucket"]
            .astype(str)
            .map(bucket_map)
            .fillna(plot_df["bucket"].astype(str))
        )
    else:
        bucket_order_fmt = bucket_order_raw
        plot_df["bucket"] = plot_df["bucket"].astype(str)

    # -----------------------------
    # Orden explícito de niveles
    # -----------------------------
    levels = sorted(
        plot_df[breakdown_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    levels_wo_total = [lvl for lvl in levels if lvl != "Total"]
    levels_final = levels_wo_total + (["Total"] if "Total" in levels else [])

    # -----------------------------
    # Colores consistentes
    # -----------------------------
    colors = [
        "#783DBE",
        "#5B8FF9",
        "#61DDAA",
        "#65789B",
        "#F6BD16",
        "#7262FD",
        "#78D3F8",
        "#9661BC",
        "#F6903D",
        "#008685",
    ]

    color_map = {
        lvl: colors[i % len(colors)]
        for i, lvl in enumerate(levels_wo_total)
    }

    if "Total" in levels_final:
        color_map["Total"] = "#555555"

    fig = go.Figure()

    for lvl in levels_final:
        sub = plot_df[plot_df[breakdown_col] == lvl].copy()

        sub["bucket"] = pd.Categorical(
            sub["bucket"],
            categories=bucket_order_fmt,
            ordered=True,
        )
        sub = sub.sort_values("bucket")

        fig.add_trace(
            go.Scatter(
                x=sub["bucket"],
                y=sub["pct"],
                mode="lines",
                name=str(lvl),
                line=dict(
                    width=3 if lvl == "Total" else 2,
                    color=color_map[lvl],
                    dash="dash" if lvl == "Total" else "solid",
                ),
                hovertemplate=(
                    f"{breakdown_col}: %{{fullData.name}}<br>"
                    "Cosecha: %{x}<br>"
                    "%{y:.2%}<extra></extra>"
                ),
            )
        )

    layout_kwargs = dict(
        height=650,
        xaxis_title="Cosecha",
        yaxis_title=build_y_label(tipo_mora, metric_mode_),
        yaxis_tickformat=".0%",
        legend_title=breakdown_col,
        hovermode="x unified",
        legend=dict(
            traceorder="normal"
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)",
            bordercolor=ATRIA_PURPLE,
            font=dict(
                size=12,
                color="#1f1f1f"
            )
        ),
        margin=dict(t=20, r=40, l=40, b=40),
    )

    if fixed_y_0_100:
        layout_kwargs["yaxis"] = dict(
            range=[0, 1],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#D9D9D9",
        )
    else:
        layout_kwargs["yaxis"] = dict(
            tickformat=".0%",
            showgrid=True,
            gridcolor="#D9D9D9",
            autorange=True,
        )

    if title:
        layout_kwargs["title"] = dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=20, color="#1f1f1f"),
        )
        layout_kwargs["margin"] = dict(t=70, r=40, l=40, b=40)

    fig.update_layout(**layout_kwargs)

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=bucket_order_fmt,
        tickangle=315
    )

    fig = apply_axis_style(fig)

    return fig

def plot_transversal_trends_matplotlib(
    df_plot,
    bucket_order,
    breakdown_col,
    title,
    tipo_mora,
    metric_mode_,
    show_point_labels: bool = True,
    force_y_0_100: bool = False,
):
    plot_df = df_plot.copy()
    plot_df[breakdown_col] = plot_df[breakdown_col].astype(str)

    # -----------------------------
    # Normalización de buckets a YYYY-MM si son fecha
    # -----------------------------
    bucket_order_raw = list(bucket_order)

    parsed_order = pd.to_datetime(bucket_order_raw, format="%Y-%m", errors="coerce")
    if pd.notna(parsed_order).all():
        bucket_order_fmt = [d.strftime("%Y-%m") for d in parsed_order]
        bucket_map = {
            str(raw): fmt for raw, fmt in zip(bucket_order_raw, bucket_order_fmt)
        }
        plot_df["bucket"] = plot_df["bucket"].astype(str).map(bucket_map).fillna(plot_df["bucket"].astype(str))
    else:
        bucket_order_fmt = [str(x) for x in bucket_order_raw]
        plot_df["bucket"] = plot_df["bucket"].astype(str)

    pivot = plot_df.pivot_table(
        index="bucket",
        columns=breakdown_col,
        values="pct",
        fill_value=np.nan
    )

    pivot = pivot.reindex(bucket_order_fmt)

    fig, ax = plt.subplots(figsize=(12, 11))

    # Colores default tipo Plotly
    colors = [
        "#783DBE",
        "#5B8FF9",
        "#61DDAA",
        "#65789B",
        "#F6BD16",
        "#7262FD",
        "#78D3F8",
        "#9661BC",
        "#F6903D",
        "#008685",
    ]

    cols = list(pivot.columns)
    cols_wo_total = [c for c in cols if c != "Total"]

    color_map = {
        col: colors[i % len(colors)]
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
                    yi + 0.002,
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
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_xlabel("Cosecha")
    ax.set_ylabel(build_y_label(tipo_mora, metric_mode_))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # solo grid horizontal estilo Excel
    ax.grid(False)
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=1.0)

    if force_y_0_100:
        ax.set_ylim(0, 1)

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

def get_sorted_cohort_labels(series: pd.Series) -> list[str]:
    """
    Devuelve cohortes válidas en formato YYYY-MM, ordenadas cronológicamente.
    """
    s = (
        series.astype(str)
        .str.strip()
        .replace({
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "<NA>": np.nan,
            "": np.nan,
        })
        .dropna()
    )

    if s.empty:
        return []

    dt = pd.to_datetime(s, format="%Y-%m", errors="coerce")
    out = (
        pd.DataFrame({"dt": dt})
        .dropna()
        .assign(label=lambda x: x["dt"].dt.strftime("%Y-%m"))
        .drop_duplicates(subset=["label"])
        .sort_values("dt")["label"]
        .tolist()
    )

    return out

def build_ui_signature(
    *,
    tipo_mora: str,
    metric_mode: str,
    filters: dict,
    castigo_enabled: bool,
    show_heatmap: bool,
    mob_max_table_ui: int,
    mob_max_line_ui: int,
    breakdown_col,
    cohort_start: str,
    cohort_end: str,
    freq_mode: str,
    value_mode: str,
    mob_fix_mult_6_only: bool,
    trans_mob_fix: int,
    trans_yaxis_0_100: bool,
) -> dict:
    """
    Firma completa de controles que sí pueden existir antes del bloque de ejecución
    y que deben disparar actualización automática del escenario.
    """

    filters_norm = {
        str(k): sorted([str(v) for v in vals])
        for k, vals in (filters or {}).items()
    }

    return {
        "tipo_mora": str(tipo_mora),
        "metric_mode": str(metric_mode),
        "filters": filters_norm,
        "castigo_enabled": bool(castigo_enabled),
        "show_heatmap": bool(show_heatmap),
        "mob_max_table_ui": int(mob_max_table_ui),
        "mob_max_line_ui": int(mob_max_line_ui),
        "breakdown_col": None if breakdown_col is None else str(breakdown_col),
        "cohort_start": str(cohort_start),
        "cohort_end": str(cohort_end),
        "freq_mode": str(freq_mode),
        "value_mode": str(value_mode),
        "mob_fix_mult_6_only": bool(mob_fix_mult_6_only),
        "trans_mob_fix": int(trans_mob_fix),
        "trans_yaxis_0_100": bool(trans_yaxis_0_100),
    }

def apply_axis_style(fig):

    fig.update_layout(
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#1f1f1f"
        ),
        xaxis=dict(
            title=dict(font=dict(size=14, color="#000000")),
            tickfont=dict(size=13, color="#000000")
        ),
        yaxis=dict(
            title=dict(font=dict(size=14, color="#000000")),
            tickfont=dict(size=13, color="#000000")
        )
    )

    return fig


# =================================
# Panel de Configuración
# =================================
st.set_page_config(page_title="Demo Matriz de Impago", layout="wide")

# --------------------------------------------------
# Cache data + engine
# --------------------------------------------------
@st.cache_data
def get_df() -> pd.DataFrame:
    return load_data()

df = get_df()

@st.cache_resource
def get_engine(df_: pd.DataFrame) -> ImpagoOnDemandEngine:
    return ImpagoOnDemandEngine(df_)

engine = get_engine(df)

# --------------------------------------------------
# Session state defaults
# --------------------------------------------------

if "last_res" not in st.session_state:
    st.session_state["last_res"] = None

if "last_sc" not in st.session_state:
    st.session_state["last_sc"] = None

if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None

if "last_ui_signature" not in st.session_state:
    st.session_state["last_ui_signature"] = None

if "live_mode_after_run" not in st.session_state:
    st.session_state["live_mode_after_run"] = False

if "reset_epoch" not in st.session_state:
    st.session_state["reset_epoch"] = 0

# --------------------------------------------------
# Sidebar – Controles de escenario
# --------------------------------------------------
epoch = st.session_state["reset_epoch"]


# =================================
# Sidebar / Configuración general
# =================================
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

# --------------------------------------------------
# Filtros
# --------------------------------------------------
st.sidebar.markdown("#### Filtros")

FILTERABLE_COLS = [
    "Perfil", "Ciudad Atria", "Plazo", "Antiguedad Unidad",
    "Laboratorios", "Notificaciones", "Atria Plus",
    "MMX 6", "MMX 12", "MMX 24",
    "Cluster Mora", "Cluster Experiencia",
    "Score generico de bancos", "Score Buro",
]

def col_has_real_values(series: pd.Series) -> bool:
    s = series.copy()
    s = s.replace(r"^\s*$", np.nan, regex=True)
    s = s.astype("object").replace({
        "nan": np.nan,
        "NaN": np.nan,
        "None": np.nan,
        "<NA>": np.nan,
        "": np.nan,
    })
    return s.notna().any()

available_cols = [
    c for c in FILTERABLE_COLS
    if c in df.columns and col_has_real_values(df[c])
]

selected_filter_cols = st.sidebar.multiselect(
    "Dimensiones a filtrar",
    options=available_cols,
    default=[],
    key=f"selected_filter_cols_ui_{epoch}",
)

filters: dict[str, list[str]] = {}
for col in selected_filter_cols:
    vals = (
        df[col]
        .replace(r"^\s*$", np.nan, regex=True)
        .dropna()
        .astype(str)
        .str.strip()
    )

    vals = sorted(v for v in vals.unique().tolist() if v.lower() != "nan")

    sel = st.sidebar.multiselect(
        f"{col}",
        options=vals,
        default=[],
        key=f"filter_vals_{col}_{epoch}",
    )

    if sel:
        filters[col] = sel

# --------------------------------------------------
# Cap MOB (común)
# --------------------------------------------------
mob_max_cap = int(pd.to_numeric(df["MOB"], errors="coerce").max())

key_heat = f"show_heatmap_ui_{epoch}"
key_mob_table = f"mob_max_table_ui_{epoch}"
key_mob_line = f"mob_max_line_ui_{epoch}"
key_break = f"breakdown_col_ui_{epoch}"

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

# --------------------------------------------------
# Totales del escenario en sidebar
# --------------------------------------------------
df_sidebar_count = engine.apply_filters(df, filters).copy()
df_sidebar_count = engine.apply_castigo_filter(df_sidebar_count)

tipo_mora_sidebar = tipo_mora  # usa la mora seleccionada actualmente en sidebar

# Normalización defensiva
if tipo_mora_sidebar in df_sidebar_count.columns:
    df_sidebar_count[tipo_mora_sidebar] = pd.to_numeric(
        df_sidebar_count[tipo_mora_sidebar], errors="coerce"
    ).fillna(0)
else:
    df_sidebar_count[tipo_mora_sidebar] = 0

if "Saldo Capital" in df_sidebar_count.columns:
    df_sidebar_count["Saldo Capital"] = pd.to_numeric(
        df_sidebar_count["Saldo Capital"], errors="coerce"
    ).fillna(0)
else:
    df_sidebar_count["Saldo Capital"] = 0.0

if "MOB" in df_sidebar_count.columns:
    df_sidebar_count["MOB"] = pd.to_numeric(
        df_sidebar_count["MOB"], errors="coerce"
    )
else:
    df_sidebar_count["MOB"] = np.nan

# --------------------------------------------------
# Folios contemplados = folios únicos que sí tienen mora > 0
# bajo los filtros del escenario
# --------------------------------------------------
df_sidebar_mora = df_sidebar_count.loc[
    df_sidebar_count[tipo_mora_sidebar] > 0
].copy()

total_folios_escenario = (
    df_sidebar_mora["folio"].nunique()
    if "folio" in df_sidebar_mora.columns and not df_sidebar_mora.empty
    else 0
)

# --------------------------------------------------
# Saldo Capital contemplado = un solo registro por folio
# usando el ÚLTIMO MOB observado con esa mora
# --------------------------------------------------
if {"folio", "Saldo Capital", "MOB"}.issubset(df_sidebar_mora.columns) and not df_sidebar_mora.empty:
    saldo_por_folio = (
        df_sidebar_mora[["folio", "MOB", "Saldo Capital"]]
        .copy()
        .sort_values(["folio", "MOB"])
        .drop_duplicates(subset=["folio"], keep="last")
    )

    total_saldo_capital_escenario = float(saldo_por_folio["Saldo Capital"].sum())
else:
    total_saldo_capital_escenario = 0.0

card_style = """
    padding: 10px 12px;
    border-left: 4px solid #783DBE;
    background-color: rgba(0, 0, 0, 0.03);
    border-radius: 6px;
    margin-top: 8px;
    margin-bottom: 4px;
"""

label_style = """
    font-size: 0.82rem;
    color: #6b7280;
"""

value_style = """
    font-size: 1.2rem;
    font-weight: 700;
    color: #1f1f1f;
"""

st.sidebar.markdown(
    f"""
    <div style="{card_style}">
        <div style="{label_style}">
            Folios contemplados en el escenario
        </div>
        <div style="{value_style}">
            {total_folios_escenario:,}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <div style="{card_style}">
        <div style="{label_style}">
            Saldo a capital contemplado en el escenario
        </div>
        <div style="{value_style}">
            ${total_saldo_capital_escenario:,.0f}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================
# Separador
# =============================
st.sidebar.divider()


# ******************************************
# Default dinámico para breakdown_col
# ******************************************
breakdown_options = ["(Ninguno)"] + available_cols

# Antes de run scenario
if len(selected_filter_cols) == 0:
    default_breakdown_ui = "(Ninguno)"

# Toma la primera dimensión elegida para filtrar
else:
    default_breakdown_ui = selected_filter_cols[0]

# Si por alguna razón no está en opciones, cae a "(Ninguno)"
if default_breakdown_ui not in breakdown_options:
    default_breakdown_ui = "(Ninguno)"

# Clave auxiliar para recordar la última configuración de filtros
key_break_sync = f"{key_break}_last_filter_cols"

current_filter_signature = tuple(selected_filter_cols)
last_filter_signature = st.session_state.get(key_break_sync)

# Empujar el valor por default al widget SOLO si:
# 1. el widget aún no tiene valor
# 2. el valor actual ya no es válido
# 3. cambió la lista de dimensiones filtradas
current_break_val = st.session_state.get(key_break)

if (
    current_break_val is None
    or current_break_val not in breakdown_options
    or current_filter_signature != last_filter_signature
    ):
    st.session_state[key_break] = default_breakdown_ui
    st.session_state[key_break_sync] = current_filter_signature


# =================================
# Sidebar / Detalles gráfico
# =================================
st.sidebar.header("Escenario / Detalle gráfico")

breakdown_col = st.sidebar.selectbox(
    "Dimensión de detalle",
    options=breakdown_options,
    key=key_break,
)

breakdown_col = None if breakdown_col == "(Ninguno)" else breakdown_col

# Mínimo interno fijo para breakdowns
min_folios_breakdown = 1

st.sidebar.markdown("### Gráficos - Análisis Prospectivo")

# =========================================================
# Base filtrada para poblar cohortes y MOBs disponibles
# =========================================================
df_controls = engine.apply_filters(df, filters).copy()
df_controls = engine.apply_castigo_filter(df_controls)

available_cohorts = get_sorted_cohort_labels(df_controls["cosecha"]) if "cosecha" in df_controls.columns else []

if not available_cohorts:
    available_cohorts = get_sorted_cohort_labels(df["cosecha"]) if "cosecha" in df.columns else []

if not available_cohorts:
    available_cohorts = ["2022-01"]

key_cohort_start = f"cohort_start_ui_{epoch}"
key_cohort_end = f"cohort_end_ui_{epoch}"

# defaults defensivos
default_cohort_start = available_cohorts[0]
default_cohort_end = available_cohorts[-1]

current_start = st.session_state.get(key_cohort_start)
current_end = st.session_state.get(key_cohort_end)

if current_start not in available_cohorts:
    st.session_state[key_cohort_start] = default_cohort_start

if current_end not in available_cohorts:
    st.session_state[key_cohort_end] = default_cohort_end

cohort_start = st.sidebar.selectbox(
    "Cosecha inicio",
    options=available_cohorts,
    key=key_cohort_start,
)

cohort_end = st.sidebar.selectbox(
    "Cosecha fin",
    options=available_cohorts,
    key=key_cohort_end,
)

# normalización defensiva por si el usuario deja inicio > fin
idx_start = available_cohorts.index(cohort_start)
idx_end = available_cohorts.index(cohort_end)

if idx_start > idx_end:
    cohort_start, cohort_end = cohort_end, cohort_start

freq_mode = st.sidebar.radio(
    "Frecuencia de cosecha",
    ["Mensual", "Trimestral"],
    key=f"prospect_freq_ui_{epoch}",
)

value_mode = st.sidebar.radio(
    "Modo de composición de barras",
    ["Monto fondeado", "Conteo de folios"],
    key=f"stack_value_mode_ui_{epoch}",
)

# =========================================================
# MOB fijo: múltiplos de 6 o cualquier MOB disponible
# =========================================================
mob_values_available = (
    pd.to_numeric(df_controls["MOB"], errors="coerce")
    .dropna()
    .astype(int)
    .sort_values()
    .unique()
    .tolist()
) if "MOB" in df_controls.columns else []

mob_values_available = [m for m in mob_values_available if 1 <= m <= mob_max_cap]

if not mob_values_available:
    mob_values_available = list(range(1, mob_max_cap + 1))

key_mult6 = f"mob_fix_mult6_only_ui_{epoch}"
key_trans_mob = f"trans_mob_fix_ui_{epoch}"

if key_mult6 not in st.session_state:
    st.session_state[key_mult6] = True

mob_fix_mult_6_only = st.session_state[key_mult6]

# Construcción de opciones antes del render
if mob_fix_mult_6_only:
    trans_mob_options = [m for m in mob_values_available if m % 6 == 0]
    if not trans_mob_options:
        trans_mob_options = mob_values_available
else:
    trans_mob_options = mob_values_available

default_trans_mob = 12 if 12 in trans_mob_options else trans_mob_options[-1]

current_trans_mob = st.session_state.get(key_trans_mob)

if current_trans_mob not in trans_mob_options:
    st.session_state[key_trans_mob] = default_trans_mob


trans_mob_fix = st.sidebar.select_slider(
    "MOB fijo",
    options=trans_mob_options,
    key=key_trans_mob,
)

st.sidebar.checkbox(
    "Usar solo múltiplos de 6",
    key=key_mult6,
)

key_trans_yaxis = f"trans_yaxis_0_100_ui_{epoch}"

trans_yaxis_0_100 = st.sidebar.checkbox(
    "Fijar eje a 100%",
    value=False,
    key=key_trans_yaxis,
)

# Leer valores anteriores 
show_heatmap = bool(st.session_state.get(key_heat, True))
mob_max_table_ui = int(st.session_state.get(key_mob_table, min(36, mob_max_cap)))
mob_max_line_ui = int(st.session_state.get(key_mob_line, min(24, mob_max_cap)))

# =============================
# Botones
# =============================

# -----------------------------
# Botón Run
# -----------------------------
run = st.sidebar.button("Run scenario", type="primary")

# -----------------------------
# Botón de Reset
# -----------------------------
def request_reset():
    old = st.session_state["reset_epoch"]

    # borra keys del epoch viejo
    for k in list(st.session_state.keys()):
        if k.endswith(f"_{old}") and (
            k.startswith((
                "tipo_mora_ui_",
                "metric_mode_ui_",
                "castigo_enabled_ui_",
                "selected_filter_cols_ui_",
                "breakdown_col_ui_",
                "show_heatmap_ui_",
                "mob_max_table_ui_",
                "mob_max_line_ui_",
                "filter_vals_",
                "cohort_start_ui_",
                "cohort_end_ui_",
                "prospect_freq_ui_",
                "stack_value_mode_ui_",
                "detail_render_mode_ui_",
                "mob_fix_mult6_only_ui_",
                "trans_mob_fix_ui_",
                "trans_yaxis_0_100_ui_",
                "stack_render_ui_",
                "trans_render_mode_ui_",
                "trans_freq_mode_ui_",
                "trans_n_cosechas_ui_",
                "n_cosechas_ui_",
            ))
        ):
            st.session_state.pop(k, None)

    st.session_state["reset_epoch"] = old + 1

    for k in [
        "last_res",
        "last_sc",
        "last_inputs",
        "last_ui_signature",
        "live_mode_after_run",
    ]:
        st.session_state.pop(k, None)

st.sidebar.button(
    "Reset",
    type="secondary",
    help="Limpia resultados y regresa a valores por defecto.",
    on_click=request_reset,
)

# Firma dinámica
current_ui_signature = build_ui_signature(
    tipo_mora=tipo_mora,
    metric_mode=metric_mode_engine,
    filters=filters,
    castigo_enabled=bool(castigo_enabled),
    show_heatmap=bool(show_heatmap),
    mob_max_table_ui=int(mob_max_table_ui),
    mob_max_line_ui=int(mob_max_line_ui),
    breakdown_col=breakdown_col,
    cohort_start=cohort_start,
    cohort_end=cohort_end,
    freq_mode=freq_mode,
    value_mode=value_mode,
    mob_fix_mult_6_only=bool(mob_fix_mult_6_only),
    trans_mob_fix=int(trans_mob_fix),
    trans_yaxis_0_100=bool(trans_yaxis_0_100),
)

# *********************************************************
# Ejecución
# 1) RUN manual activa el modo live
# 2) Después de eso, cualquier cambio en sidebar re-ejecuta
# *********************************************************

should_run = False
run_reason = None

if run:
    should_run = True
    run_reason = "manual"

elif (
    st.session_state.get("live_mode_after_run", False)
    and st.session_state.get("last_ui_signature") is not None
    and current_ui_signature != st.session_state["last_ui_signature"]
):
    should_run = True
    run_reason = "auto"

if should_run:
    sc = Scenario(
        name="streamlit_demo_v1",
        tipo_mora=tipo_mora,
        metric_mode=metric_mode_engine,
        filters=filters,
        breakdown_col=breakdown_col,
    )

    spinner_msg = (
        "Ejecutando escenario."
        if run_reason == "manual"
        else "Actualizando escenario automáticamente..."
    )

    with st.spinner(spinner_msg):
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
            "show_heatmap": bool(show_heatmap),
            "mob_max_table_ui": int(mob_max_table_ui),
            "mob_max_line_ui": int(mob_max_line_ui),
            "cohort_start": cohort_start,
            "cohort_end": cohort_end,
            "freq_mode": freq_mode,
            "value_mode": value_mode,
            "mob_fix_mult_6_only": bool(mob_fix_mult_6_only),
            "trans_mob_fix": int(trans_mob_fix),
        }
        st.session_state["last_ui_signature"] = current_ui_signature
        st.session_state["live_mode_after_run"] = True

    if run_reason == "manual":
        st.success("Listo")

# *********************************************************
# 2) RENDER -> si ya existe resultado previo
# *********************************************************
if st.session_state.get("last_res") is not None:
    res = st.session_state["last_res"]
    sc = st.session_state["last_sc"]
    inputs = st.session_state["last_inputs"]

    tipo_mora_run = inputs["tipo_mora"]
    metric_mode_run = inputs["metric_mode"]
    filters_run = inputs["filters"]
    breakdown_col_run = inputs["breakdown_col"]
    castigo_enabled_run = inputs["castigo_enabled"]

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
        tipo_mora_=sc.tipo_mora,
        metric_mode_=sc.metric_mode,
        filters_=filters,
        castigo_enabled_=bool(castigo_enabled),
        mob_max_=mob_max_table,
        breakdown_col_=None,            
        include_detail_=False,
    )

    title_line_cosechas = build_plot_title(
        tipo_mora_=sc.tipo_mora,
        metric_mode_=sc.metric_mode,
        filters_=filters,
        castigo_enabled_=bool(castigo_enabled),
        mob_max_=mob_max_line,
        breakdown_col_=None,            
        include_detail_=False,
    )

    title_line_detalle = build_plot_title(
        tipo_mora_=sc.tipo_mora,
        metric_mode_=sc.metric_mode,
        filters_=filters,
        castigo_enabled_=bool(castigo_enabled),
        mob_max_=mob_max_line,
        breakdown_col_=breakdown_col,   
        include_detail_=False,           
    )

    title_detail_prospectivo = f"{title_line_detalle}, MOB {trans_mob_fix} fijo"

    file_stub_table = slug_filename(title_table)
    file_stub_line  = slug_filename(title_line_cosechas)
    file_stub_det   = slug_filename(title_line_detalle)
    file_stub_prosp = slug_filename(title_detail_prospectivo)


    # =========================================================
    # Navegación principal persistente
    # =========================================================
    if "main_view" not in st.session_state:
        st.session_state.main_view = "Cosechas"

    selected_view = st.segmented_control(
        "Vista principal",
        options=["Cosechas", "Detalle gráfico"],
        key="main_view",
        label_visibility="collapsed"
    )

    st.divider()


    # =========================================================
    # Vista: Cosechas
    # =========================================================
    if selected_view == "Cosechas":

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
        # Curvas externas (hasta 10) - SOLO EXCEL
        # -------------------------
        st.markdown("### Comparación: curvas externas (Excel)")

        help_short = ('Sube hasta 10 archivos Excel con columnas MOB y value.\n\n'
                     'Debe ser un archivo Excel (`.xlsx` o `.xls`)\n\n'
                     'Debe contener exactamente estas columnas: `MOB`, `value`\n\n'
                     '`MOB` debe ser entero\n\n'
                     '`value` debe ser numérico\n\n'
                     f'Se recorta a `MOB <= {mob_max_line}`')
                        
        # Ingreso de gráficas por usuario
        compare_enabled = st.checkbox(
            "Agregar curvas externas",
            value=False,
            key=f"compare_enabled_{epoch}",
            help=help_short,
        )

        if compare_enabled:
            st.caption(
                    f"""
            **Ejemplo de formato requerido por archivo**
            ```text
            MOB,value
            1,0.2
            2,0.3
            ...
            24,16.1
            ```
            """
                )

        uploaded_excels = st.file_uploader(
            "Subir Excel(s)",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key=f"ext_excels_{epoch}",
            disabled=not compare_enabled,
            help=help_short,
        )
        
        external_curves = []
        if compare_enabled and uploaded_excels:
            if len(uploaded_excels) > 10:
                st.warning("Subiste más de 10 archivos. Solo se usarán los primeros 10.")
                uploaded_excels = uploaded_excels[:10]

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
            sc.metric_mode,
            tipo_mora=sc.tipo_mora,
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

        # --- Plotly ---
        if use_plotly_tab1:
            fig_line = plot_curve_agg_plotly(
                curve_view,
                sc.metric_mode,
                tipo_mora=sc.tipo_mora,
                mob_max=mob_max_line,
                title=title_line_cosechas,
                external_curves=external_curves if (compare_enabled and external_curves) else None,
            )
            st.plotly_chart(fig_line, width="stretch")

        # Matplotlib con etiquetas
        else:
            
            fig_line = plot_curve_agg(
                curve_view,
                sc.metric_mode,
                tipo_mora=sc.tipo_mora,
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


    # =========================================================
    # Vista: Detalle gráfico
    # =========================================================
    elif selected_view == "Detalle gráfico":

        # Spinner
        if breakdown_col is None:
            st.info("Selecciona una columna en 'Detalle de gráfico' para ver curvas por nivel.")

        # Renders
        else:

            # -------------------------
            # Plot lineplots
            # -------------------------

            render_mode = st.radio(
                "Modo de gráfica",
                ["Interactivo", "Con etiquetas %"],
                index=0,
                horizontal=True,
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

            # =========================================================
            # Barras apiladas + Tendencias transversales lado a lado
            # =========================================================
            detail_render_mode = st.radio(
                "Modo de gráficas prospectivas",
                ["Interactivo", "Con etiquetas"],
                index=0,
                horizontal=True,
                key=f"detail_render_mode_ui_{epoch}",
            )
            
            st.markdown(
                f"""
                <div style="
                    text-align: center;
                    font-size: 1.8rem;
                    font-weight: 900;
                    color: #1f1f1f;
                    margin-top: 0;
                    margin-bottom: -0.1rem;
                ">
                    {title_detail_prospectivo}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_stack, col_trans = st.columns([1, 1.15])

            # ----------------------------------------------
            # COLUMNA IZQUIERDA: Barras apiladas
            # ----------------------------------------------
            with col_stack:

                agg, bucket_order = engine.compute_breakdown_composition(
                    scenario=sc,
                    breakdown_col=breakdown_col,
                    freq_mode=freq_mode,
                    value_mode=value_mode,
                    cohort_start=cohort_start,
                    cohort_end=cohort_end,
                )

                if agg.empty or len(bucket_order) == 0:
                    st.info("No hay datos disponibles para la composición en el rango de cohortes seleccionado.")
                
                else:

                    # --- Plotly ---
                    if detail_render_mode == "Interactivo":

                        fig = plot_stacked_plotly(
                            agg,
                            bucket_order,
                            breakdown_col,
                            ""
                        )

                        st.plotly_chart(fig, width="stretch")

                        file_bytes = fig.to_html().encode()
                        filename = f"{file_stub_prosp}_barras_apiladas.html"
                        mime = "text/html"

                    # --- Matplotlib ---
                    else:

                        fig = plot_stacked_matplotlib(
                            agg,
                            bucket_order,
                            breakdown_col,
                            ""
                        )

                        st.pyplot(fig)

                        file_bytes = fig_to_png_bytes(fig)
                        filename = f"{file_stub_prosp}_barras_apiladas.png"
                        mime = "image/png"

                    st.download_button(
                        "Descargar gráfica",
                        data=file_bytes,
                        file_name=filename,
                        mime=mime,
                        key=f"download_stack_{epoch}",
                        use_container_width=True,
                    )

            # ----------------------------------------------
            # COLUMNA DERECHA: Tendencias transversales
            # ----------------------------------------------
            with col_trans:

                trans_df, trans_bucket_order = engine.compute_breakdown_transversal_trends(
                    scenario=sc,
                    breakdown_col=breakdown_col,
                    mob_fix=trans_mob_fix,
                    freq_mode=freq_mode,
                    cohort_start=cohort_start,
                    cohort_end=cohort_end,
                )

                trans_title = ""

                if trans_df.empty or len(trans_bucket_order) == 0:
                    st.info(
                        "No hay datos disponibles para tendencias transversales con ese rango de cohortes "
                        "y el MOB fijo seleccionado. Prueba con un MOB menor o un rango más antiguo."
                    )

                else:

                    # --- Plotly ---
                    if detail_render_mode == "Interactivo":

                        fig = plot_transversal_trends_plotly(
                            trans_df,
                            trans_bucket_order,
                            breakdown_col,
                            trans_title,
                            tipo_mora=sc.tipo_mora,
                            metric_mode_=sc.metric_mode,
                            fixed_y_0_100 = trans_yaxis_0_100
                        )

                        st.plotly_chart(fig, width="stretch")

                        file_bytes = fig.to_html().encode()
                        filename = f"{file_stub_prosp}_tendencias_transversales.html"
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
                            force_y_0_100=trans_yaxis_0_100
                        )

                        st.pyplot(fig)

                        file_bytes = fig_to_png_bytes(fig)
                        filename = f"{file_stub_prosp}_tendencias_transversales.png"
                        mime = "image/png"

                    st.download_button(
                        "Descargar gráfica",
                        data=file_bytes,
                        file_name=filename,
                        mime=mime,
                        key=f"download_transversal_{epoch}",
                        use_container_width=True,
                    )


# -----------------------------
# Spinner
# -----------------------------
else:
    st.info("Configura el escenario en el panel izquierdo y presiona **Run scenario**.")
