from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
from typing import Optional, Union, Literal

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go


@dataclass
class HarvestConfig:
    # Columnas base
    fecha_inicio_col: str = "Fecha de Inicio"
    mob_col: str = "MOB"
    cohorte_col: str = "cohorte"

    # --- Modo de métrica ---
    # exposure: %impago = sum(BGI) / sum(Monto Fondeado) * 100
    # ever:     %ever  = sum(EVER) / denom_folios * 100
    metric_mode: Literal["exposure", "ever"] = "exposure"

    # Exposure (exposición)
    type_bgi: str = "BGI4+"          # e.g., "BGI4+"
    monto_fond: str = "Monto Fondeado"

    # Ever (bandera)
    ever_col: str = "BGI4+_CONTEO_EVER"  # e.g., "BGI3+_CONTEO_EVER"
    folio_col: str = "folio"

    # Denominador para ever:
    # - 'variable': por cohorte×MOB (folios observados en ese MOB) [DEFAULT]
    # - 'fixed':    fijo por cohorte (folios originados en ese cohorte)
    ever_den_mode: Literal["fixed", "variable"] = "variable"

    # Reglas
    min_saldo: float = 0.0   # exposure: si monto_sum <= min_saldo => NaN
    min_den: float = 0.0     # ever: si denom <= min_den => NaN
    percent_scale: float = 100.0  # 100 para porcentaje, 1 para razón


class HarvestImpagoMatrix:

    """
    Calcula la matriz de % impago (exposición) por cohorte × MOB:

        %impago = sum(BGI) / sum(Monto Fondeado) * 100

    Devuelve:
        - agg (long): cohorte, MOB, bgi_sum, monto_sum, pct_impago
        - matriz (wide): index=cohorte, columns=MOB
    """

    # Constructor ----------------------------------------------------------------------------
    def __init__(self, 
                 df: pd.DataFrame, 
                 config: Optional[HarvestConfig] = None,
                 vect_comp: bool = False):
        self.df = df.copy()
        self.config = config or HarvestConfig()
        self.agg_: Optional[pd.DataFrame] = None
        self.matriz_: Optional[pd.DataFrame] = None
        self.vect_comp = vect_comp
        self.plot_label = self.config.type_bgi if self.config.metric_mode == 'exposure' else self.config.ever_col


    # Carga de archivos ----------------------------------------------------------------------
    @classmethod
    def from_excel(
        cls,
        path: str,
        sheet_name: Union[str, int] = 0,
        config: Optional[HarvestConfig] = None,
        **read_excel_kwargs
    ) -> "HarvestImpagoMatrix":
        df = pd.read_excel(path, sheet_name=sheet_name, **read_excel_kwargs)
        return cls(df=df, config=config)


    # Helpers --------------------------------------------------------------------------------
    def _metric_value_col(self) -> str:
        c = self.config
        return "pct_impago" if c.metric_mode == "exposure" else "pct_ever"

    def _metric_label(self) -> str:
        c = self.config
        if c.metric_mode == "exposure":
            return f"% Impago ({c.type_bgi} / {c.monto_fond})"
        return f"% Ever ({c.ever_col} / #folios)"


    # Función de cálculo de métricas ----------------------------------------------------------
    def calc_pct_ratio(
        self,
        data: Union[pd.DataFrame, pd.Series, dict],
        *,
        mode: Literal["raw", "agg"] = "raw",
        num_col: str,
        den_col: str,
        num_sum_col: str,
        den_sum_col: str,
        min_den: float = 0.0,
        percent_scale: float = 100.0,
        out_name: str = "pct",
    ):
        """Single source of truth para métricas tipo ratio (num/den * scale).

        - raw: data es df crudo -> float
        - agg: data es df agregado -> Series (vectorizado si self.vect_comp)
               o fila agregada (Series/dict) -> float

        Regla: si denominador <= min_den -> NaN
        """

        if mode == "raw":
            den_total = pd.to_numeric(data[den_col], errors="coerce").sum(skipna=True)
            if den_total <= min_den:
                return np.nan
            num_total = pd.to_numeric(data[num_col], errors="coerce").sum(skipna=True)
            return (num_total / den_total) * percent_scale

        if mode == "agg":
            # DataFrame agregado
            if isinstance(data, pd.DataFrame):
                den = pd.to_numeric(data[den_sum_col], errors="coerce")
                num = pd.to_numeric(data[num_sum_col], errors="coerce")

                if self.vect_comp:
                    out = np.full(len(data), np.nan, dtype=float)
                    mask = den > min_den
                    out[mask.to_numpy()] = (num[mask] / den[mask]) * percent_scale
                    return pd.Series(out, index=data.index, name=out_name)

                out_list = []
                for _, row in data.iterrows():
                    den_i = float(row[den_sum_col])
                    if den_i <= min_den:
                        out_list.append(np.nan)
                    else:
                        out_list.append((float(row[num_sum_col]) / den_i) * percent_scale)
                return pd.Series(out_list, index=data.index, name=out_name)

            # Fila agregada
            if isinstance(data, dict):
                data = pd.Series(data)

            den_total = float(data[den_sum_col])
            if den_total <= min_den:
                return np.nan
            return (float(data[num_sum_col]) / den_total) * percent_scale

        raise ValueError("mode debe ser 'raw' o 'agg'")

    def calc_pct_impago(
        self,
        data: Union[pd.DataFrame, pd.Series, dict],
        *,
        mode: Literal["raw", "agg"] = "raw",
        bgi_col: Optional[str] = None,
        monto_fond: Optional[str] = None,
        bgi_sum_col: str = "bgi_sum",
        monto_sum_col: str = "monto_sum",
    ):
        """Calcula % impago (exposición): sum(BGI)/sum(Monto)*100.

        Nota: se mantiene por compatibilidad; la lógica real vive en calc_pct_ratio.
        """
        c = self.config
        bgi_col = bgi_col or c.type_bgi
        monto_fond = monto_fond or c.monto_fond
        return self.calc_pct_ratio(
            data,
            mode=mode,
            num_col=bgi_col,
            den_col=monto_fond,
            num_sum_col=bgi_sum_col,
            den_sum_col=monto_sum_col,
            min_den=getattr(c, "min_saldo", 0.0),
            percent_scale=getattr(c, "percent_scale", 100.0),
            out_name="pct_impago",
        )


    # Cálculo de la matriz triangular ------------------------------------------------------------
    def add_cohorte(self) -> "HarvestImpagoMatrix":
        c = self.config
        if c.fecha_inicio_col not in self.df.columns:
            raise ValueError(f"No existe columna '{c.fecha_inicio_col}' en df.")
        self.df[c.fecha_inicio_col] = pd.to_datetime(self.df[c.fecha_inicio_col], errors="coerce")
        self.df[c.cohorte_col] = self.df[c.fecha_inicio_col].dt.to_period("M").dt.to_timestamp()
        return self

    def validate_required_columns(self) -> "HarvestImpagoMatrix":
        c = self.config
        if c.metric_mode == "exposure":
            required = {c.cohorte_col, c.mob_col, c.type_bgi, c.monto_fond}
        elif c.metric_mode == "ever":
            required = {c.cohorte_col, c.mob_col, c.ever_col, c.folio_col}
        else:
            raise ValueError("metric_mode debe ser 'exposure' o 'ever'")

        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
        return self

    def compute_agg(self) -> "HarvestImpagoMatrix":
        c = self.config
        self.validate_required_columns()

        if c.metric_mode == "exposure":
            agg = (
                self.df.groupby([c.cohorte_col, c.mob_col], as_index=False)
                .agg(
                    bgi_sum=(c.type_bgi, "sum"),
                    monto_sum=(c.monto_fond, "sum"),
                )
            )

            agg["pct_impago"] = self.calc_pct_impago(
                agg,
                mode="agg",
                bgi_sum_col="bgi_sum",
                monto_sum_col="monto_sum",
            )

            self.agg_ = agg
            return self

        # --- EVER MODE ---
        # %ever_k = sum(EVER) / denom_folios * 100
        # EVER repetido por folio×MOB (cumulativo). Denominador:
        # - variable (default): #folios observados en ese cohorte×MOB
        # - fixed:              #folios del cohorte (constante en todos los MOB)
        if c.ever_den_mode == "variable":
            agg = (
                self.df.groupby([c.cohorte_col, c.mob_col], as_index=False)
                .agg(
                    ever_sum=(c.ever_col, "sum"),
                    folio_den=(c.folio_col, "nunique"),
                )
            )
        elif c.ever_den_mode == "fixed":
            den = (
                self.df.groupby(c.cohorte_col, as_index=False)
                .agg(folio_den=(c.folio_col, "nunique"))
            )
            agg = (
                self.df.groupby([c.cohorte_col, c.mob_col], as_index=False)
                .agg(
                    ever_sum=(c.ever_col, "sum"),
                )
                .merge(den, on=c.cohorte_col, how="left")
            )
        else:
            raise ValueError("ever_den_mode debe ser 'variable' o 'fixed'")

        agg["pct_ever"] = self.calc_pct_ratio(
            agg,
            mode="agg",
            num_col=c.ever_col,
            den_col=c.folio_col,
            num_sum_col="ever_sum",
            den_sum_col="folio_den",
            min_den=getattr(c, "min_den", 0.0),
            percent_scale=getattr(c, "percent_scale", 100.0),
            out_name="pct_ever",
        )

        self.agg_ = agg
        return self

    def compute_matrix(self) -> "HarvestImpagoMatrix":
        c = self.config
        if self.agg_ is None:
            self.compute_agg()

        value_col = self._metric_value_col()

        matriz = (
            self.agg_
            .pivot(index=c.cohorte_col, columns=c.mob_col, values=value_col)
            .sort_index()
        )

        matriz.index.name = c.cohorte_col
        matriz.columns.name = c.mob_col

        self.matriz_ = matriz
        return self

    def run(self) -> "HarvestImpagoMatrix":
        self.add_cohorte()
        self.compute_matrix()
        return self


    # Getters ------------------------------------------------------------------------------
    def get_agg(self) -> pd.DataFrame:
        if self.agg_ is None:
            raise ValueError("Aún no se ha calculado agg_. Ejecuta .run() o .compute_agg().")
        return self.agg_

    def get_matrix(self) -> pd.DataFrame:
        if self.matriz_ is None:
            raise ValueError("Aún no se ha calculado matriz_. Ejecuta .run() o .compute_matrix().")
        return self.matriz_


    # Guardado de matriz calculada ----------------------------------------------------------
    def save_matrix(
        self,
        path: str,
        fmt: Literal["excel", "parquet", "csv"] = "excel"
    ) -> None:
        
        m = self.get_matrix().copy()

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if fmt == "excel":
            # Excel no preserva columns.name siempre, pero esto ayuda.
            m.to_excel(path, merge_cells=False)
            print(f"Archivo guardado en: {path}")
        elif fmt == "parquet":
            m.to_parquet(path)
            print(f"Archivo guardado en: {path}")
        elif fmt == "csv":
            m.to_csv(path)
            print(f"Archivo guardado en: {path}")
        else:
            raise ValueError("fmt debe ser: 'excel', 'parquet' o 'csv'.")

    def save_agg(
        self,
        path: str,
        fmt: Literal["excel", "parquet", "csv"] = "csv"
    ) -> None:
        
        a = self.get_agg().copy()

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if fmt == "excel":
            a.to_excel(path, index=False)
            print(f"Archivo guardado en: {path}")
        elif fmt == "parquet":
            a.to_parquet(path, index=False)
            print(f"Archivo guardado en: {path}")
        elif fmt == "csv":
            a.to_csv(path, index=False)
            print(f"Archivo guardado en: {path}")
        else:
            raise ValueError("fmt debe ser: 'excel', 'parquet' o 'csv'.")


    # Visualizaciones con HeatMap -------------------------------------------------------------
    def _ensure_matrix(self) -> pd.DataFrame:
            return self.get_matrix()

    @staticmethod
    def _cohort_range(matriz: pd.DataFrame,
                      cohort_start: Optional[Union[str, pd.Timestamp]] = None,
                      cohort_end: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
        m = matriz.copy()
        if not isinstance(m.index, pd.DatetimeIndex):
            m.index = pd.to_datetime(m.index, errors="coerce")

        if cohort_start is not None:
            cohort_start = pd.to_datetime(cohort_start)
            m = m.loc[m.index >= cohort_start]

        if cohort_end is not None:
            cohort_end = pd.to_datetime(cohort_end)
            m = m.loc[m.index <= cohort_end]

        if m.empty:
            raise ValueError("No hay cohortes en el rango seleccionado.")

        return m

    def plot_heatmap_classic(
        self,
        cohort_start: Optional[Union[str, pd.Timestamp]] = None,
        cohort_end: Optional[Union[str, pd.Timestamp]] = None,
        cmap: str = "Reds",
        figsize=(14, 7.5),
        title: Optional[str] = None,
        show: bool = True,
    ):
        """
        Heatmap clásico:
        - X = MOB
        - Y = Cohorte
        - Origen arriba-izquierda (default de imshow)
        """
        matriz = self._ensure_matrix()
        m = self._cohort_range(matriz, cohort_start, cohort_end)

        plt.figure(figsize=figsize)
        plt.imshow(m, aspect="auto", cmap=cmap)
        plt.colorbar(label="% Impago")
        plt.xlabel("MOB")
        plt.ylabel("Cohorte")

        if title is None:
            if cohort_start and cohort_end:
                title = f"Matriz de Cosecha – % Impago {self.plot_label} ({pd.to_datetime(cohort_start):%Y-%m} a {pd.to_datetime(cohort_end):%Y-%m})"
            elif cohort_start:
                title = f"Matriz de Cosecha – % Impago {self.plot_label} (desde {pd.to_datetime(cohort_start):%Y-%m})"
            else:
                title = f"Matriz de Cosecha – % Impago {self.plot_label}"
        plt.title(title)

        plt.yticks(range(len(m.index)), m.index.strftime("%Y-%m"))
        plt.xticks(range(len(m.columns)), m.columns)

        plt.tight_layout()
        if show:
            plt.show()

    def plot_heatmap_vertical(
            
        self,
        cohort_start: Optional[Union[str, pd.Timestamp]] = None,
        cohort_end: Optional[Union[str, pd.Timestamp]] = None,
        cmap: str = "Reds",
        figsize=(10, 6),
        title: Optional[str] = None,
        single_cohort_figsize=(5, 7),
        show: bool = True,
    ):
        """
        Heatmap vertical:
        - X = Cohorte
        - Y = MOB
        - Origen abajo-izquierda (origin='lower')
        Ideal para ver una cohorte o un rango comparado.
        """
        matriz = self._ensure_matrix()
        m = self._cohort_range(matriz, cohort_start, cohort_end)

        # Transpuesta para MOB en Y
        m_t = m.T

        # Tamaño dinámico si es una sola cohorte
        if cohort_start is not None and cohort_end is not None and pd.to_datetime(cohort_start) == pd.to_datetime(cohort_end):
            fig = single_cohort_figsize
        else:
            fig = figsize

        plt.figure(figsize=fig)
        plt.imshow(m_t, aspect="auto", cmap=cmap, origin="lower")
        plt.colorbar(label="% Impago")
        plt.xlabel("Cohorte")
        plt.ylabel("MOB")

        if title is None:
            if cohort_start and cohort_end:
                if pd.to_datetime(cohort_start) == pd.to_datetime(cohort_end):
                    title = f"Matriz de Cosecha – % Impago {self.plot_label} ({pd.to_datetime(cohort_start):%Y-%m})"
                else:
                    title = f"Matriz de Cosecha – % Impago {self.plot_label} ({pd.to_datetime(cohort_start):%Y-%m} a {pd.to_datetime(cohort_end):%Y-%m})"
            elif cohort_start:
                title = f"Matriz de Cosecha – % Impago {self.plot_label} (desde {pd.to_datetime(cohort_start):%Y-%m} hasta la fecha)"
            else:
                title = f"Matriz de Cosecha – % Impago {self.plot_label}"
        plt.title(title)

        plt.xticks(range(len(m_t.columns)), m_t.columns.strftime("%Y-%m"), rotation=90)
        plt.yticks(range(len(m_t.index)), m_t.index)

        plt.tight_layout()
        if show:
            plt.show()


    # Visualizaciones con curvas %impago/cosecha ------------------------------------------------

    def plot_cohort_curves_all(
        self,
        cohort_start: Optional[Union[str, pd.Timestamp]] = None,
        cohort_end: Optional[Union[str, pd.Timestamp]] = None,
        show: bool = True,
        max_cohorts: Optional[int] = None,
        figsize=(10,6)
    ):
        """
        Spaghetti plot interactivo con Plotly:
        - Hover muestra cohorte, MOB y valor
        - Zoom / pan
        - Click en leyenda para ocultar/mostrar cohortes
        """

        title = f"Curvas de Cosecha por Cohorte - {self.plot_label}"

        matriz = self.get_matrix().copy()
        if not isinstance(matriz.index, pd.DatetimeIndex):
            matriz.index = pd.to_datetime(matriz.index, errors="coerce")

        # Filtros por cohorte (rango de fechas)
        if cohort_start is not None:
            cohort_start = pd.to_datetime(cohort_start)
            matriz = matriz.loc[matriz.index >= cohort_start]

        if cohort_end is not None:
            cohort_end = pd.to_datetime(cohort_end)
            matriz = matriz.loc[matriz.index <= cohort_end]

        if matriz.empty:
            raise ValueError("No hay cohortes en el rango seleccionado.")

        # Límite de cantidades de cohortes
        if max_cohorts is not None and len(matriz) > max_cohorts:
            matriz = matriz.tail(max_cohorts)

        fig = go.Figure()

        x = list(matriz.columns)

        for cohorte, row in matriz.iterrows():
            label = cohorte.strftime("%Y-%m") if pd.notna(cohorte) else "NaT"

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=row.values,
                    mode="lines",
                    name=label,
                    hovertemplate=(
                        "Cohorte: %{fullData.name}"
                        "<br>MOB: %{x}"
                        "<br>%: %{y:.2f}"
                        "<extra></extra>"
                    ),
                )
            )

        # Figsize en pixeles
        width = int(figsize[0] * 100)
        height = int(figsize[1] * 100)
        fig.update_layout(
            title=title,
            xaxis_title="MOB",
            yaxis_title=f"% Impago {self.plot_label}",
            hovermode="closest",
            legend_title_text="Cohorte",
            template="plotly_white",
            width=width, 
            height=height
        )

        # Mejoras de UX: líneas suaves, zoom
        fig.update_traces(line=dict(width=2))

        if show:
            fig.show()
            return None

        return fig

    @staticmethod
    def _rank_top_cohorts(
        matriz: pd.DataFrame,
        mode: Literal["last", "tail_mean", "mob"],
        top_n: int = 10,
        mob_k: Optional[int] = None,
        tail_n: int = 3,
    ) -> pd.Index:
        """
        Devuelve el índice (cohortes) del Top N de acuerdo a un criterio.
        """
        if mode == "last":
            score = matriz.apply(lambda r: r.dropna().iloc[-1] if r.dropna().shape[0] else float("nan"), axis=1)
        elif mode == "tail_mean":
            score = matriz.apply(lambda r: r.dropna().tail(tail_n).mean() if r.dropna().shape[0] else float("nan"), axis=1)
        elif mode == "mob":
            if mob_k is None:
                raise ValueError("Para mode='mob' debes pasar mob_k.")
            if mob_k not in matriz.columns:
                raise ValueError(f"mob_k={mob_k} no existe en columnas de la matriz.")
            score = matriz[mob_k]
        else:
            raise ValueError("mode debe ser 'last', 'tail_mean' o 'mob'.")

        score = score.dropna()
        return score.sort_values(ascending=False).head(top_n).index

    def plot_top_cohort_curves(
        self,
        mode: Literal["last", "tail_mean", "mob"] = "last",
        top_n: int = 10,
        mob_k: Optional[int] = None,
        tail_n: int = 3,
        cohort_start: Optional[Union[str, pd.Timestamp]] = None,
        cohort_end: Optional[Union[str, pd.Timestamp]] = None,
        figsize=(10, 6),
        linewidth: float = 2.5,
        background_alpha: float = 0.25,
        title: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot interactivo (Plotly): grafica todas las cohortes (fondo gris) y resalta el Top N
        según criterio (last / tail_mean / mob).

        - Hover muestra cohorte, MOB y valor
        - Click en leyenda para ocultar/mostrar
        - show=True => fig.show() y retorna None
        show=False => retorna fig
        """

        matriz = self.get_matrix().copy()
        if not isinstance(matriz.index, pd.DatetimeIndex):
            matriz.index = pd.to_datetime(matriz.index, errors="coerce")

        if cohort_start is not None:
            cohort_start = pd.to_datetime(cohort_start)
            matriz = matriz.loc[matriz.index >= cohort_start]

        if cohort_end is not None:
            cohort_end = pd.to_datetime(cohort_end)
            matriz = matriz.loc[matriz.index <= cohort_end]

        if matriz.empty:
            raise ValueError("No hay cohortes en el rango seleccionado.")

        # Top cohorts (índice de cohortes)
        top_idx = self._rank_top_cohorts(
            matriz=matriz,
            mode=mode,
            top_n=top_n,
            mob_k=mob_k,
            tail_n=tail_n,
        )
        top_set = set(top_idx)

        # Título por defecto
        if title is None:
            if mode == "last":
                title = f"Curvas de Cosecha – Top {top_n} cohortes con mayor % Impago (último MOB)"
            elif mode == "tail_mean":
                title = f"Curvas de Cosecha – Top {top_n} cohortes con mayor % Impago (promedio últimos {tail_n} MOB)"
            elif mode == "mob":
                title = f"Curvas de Cosecha – Top {top_n} cohortes (% Impago en MOB {mob_k})"

        fig = go.Figure()
        x = list(matriz.columns)

        # (1) Fondo: todas las cohortes no-top en gris
        for cohorte, row in matriz.iterrows():
            if cohorte in top_set:
                continue

            label = cohorte.strftime("%Y-%m") if pd.notna(cohorte) else "NaT"
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=row.values,
                    mode="lines",
                    name=label,          # nombre queda, pero no mostramos en leyenda
                    showlegend=False,
                    line=dict(width=1.5, color="gray"),
                    opacity=background_alpha,
                    hovertemplate=(
                        "Cohorte: %{fullData.name}"
                        "<br>MOB: %{x}"
                        "<br>%: %{y:.2f}"
                        "<extra></extra>"
                    ),
                )
            )

        # (2) Top: resaltados + sí en leyenda
        for cohorte in top_idx:
            if cohorte not in matriz.index:
                continue
            row = matriz.loc[cohorte]
            label = cohorte.strftime("%Y-%m") if pd.notna(cohorte) else "NaT"

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=row.values,
                    mode="lines",
                    name=label,
                    showlegend=True,
                    line=dict(width=linewidth),
                    hovertemplate=(
                        "Cohorte (TOP): %{fullData.name}"
                        "<br>MOB: %{x}"
                        "<br>%: %{y:.2f}"
                        "<extra></extra>"
                    ),
                )
            )

        # Mapear figsize a pixeles aprox
        width = int(figsize[0] * 100)
        height = int(figsize[1] * 100)

        fig.update_layout(
            title=title,
            xaxis_title="MOB",
            yaxis_title=f"% Impago {self.plot_label}",
            hovermode="closest",
            legend_title_text="Top Cohortes",
            template="plotly_white",
            width=width,
            height=height,
        )

        if show:
            fig.show()
            return None

        return fig

    def plot_portfolio_curve_mean(
        self,
        cohort_start: Optional[Union[str, pd.Timestamp]] = None,
        cohort_end: Optional[Union[str, pd.Timestamp]] = None,
        figsize=(10, 6),
        marker: str = "o",
        show: bool = True,
        debug: bool = False,
    ):
        """
        Evolución mensual del % impago del portafolio por mes calendario (Mes-Año).

        Interactivo con Plotly (hover, zoom, pan).
        """

        title = f"Evolución mensual del % Impago Medio del Portafolio ({self.plot_label})"

        df = self.df.copy()

        cfg = self.config
        mes_col = "Mes-Año"
        fecha_inicio_col = cfg.fecha_inicio_col
        bgi_col = cfg.type_bgi
        monto_fond = cfg.monto_fond

        # Validaciones mínimas
        required = [mes_col, fecha_inicio_col, bgi_col, monto_fond]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"plot_portfolio_curve_mean (mensual): faltan columnas {missing}. "
                f"Disponibles: {list(df.columns)}"
            )

        # Normalizar fechas
        df[mes_col] = pd.to_datetime(df[mes_col], errors="coerce")
        df[fecha_inicio_col] = pd.to_datetime(df[fecha_inicio_col], errors="coerce")

        # Cohortes (auditoría / consistencia)
        df["cohorte"] = df[fecha_inicio_col].dt.to_period("M").dt.to_timestamp()

        # Filtro por rango (mes calendario)
        if cohort_start is not None:
            cohort_start = pd.to_datetime(cohort_start, errors="coerce")
            df = df.loc[df[mes_col] >= cohort_start]

        if cohort_end is not None:
            cohort_end = pd.to_datetime(cohort_end, errors="coerce")
            df = df.loc[df[mes_col] <= cohort_end]

        if df.empty:
            raise ValueError("No hay registros en el rango seleccionado (Mes-Año).")

        c = self.config

        if c.metric_mode == "exposure":
            ts = (
                df.groupby(mes_col, as_index=True)
                .agg(
                    bgi_sum=(c.type_bgi, "sum"),
                    monto_sum=(c.monto_fond, "sum"),
                )
                .sort_index()
            )

            ts["pct_portafolio"] = self.calc_pct_impago(
                ts,
                mode="agg",
                bgi_sum_col="bgi_sum",
                monto_sum_col="monto_sum",
            )

        else:  # c.metric_mode == "ever"
            if c.ever_den_mode == "variable":
                ts = (
                    df.groupby(mes_col, as_index=True)
                    .agg(
                        ever_sum=(c.ever_col, "sum"),
                        folio_den=(c.folio_col, "nunique"),
                    )
                    .sort_index()
                )
            elif c.ever_den_mode == "fixed":
                total_den = df[c.folio_col].nunique()
                ts = (
                    df.groupby(mes_col, as_index=True)
                    .agg(
                        ever_sum=(c.ever_col, "sum"),
                    )
                    .sort_index()
                )
                ts["folio_den"] = float(total_den)
            else:
                raise ValueError("ever_den_mode debe ser 'variable' o 'fixed'")

            ts["pct_portafolio"] = self.calc_pct_ratio(
                ts,
                mode="agg",
                num_col=c.ever_col,      
                den_col=c.folio_col,     
                num_sum_col="ever_sum",
                den_sum_col="folio_den",
                min_den=getattr(c, "min_den", 0.0),
                percent_scale=getattr(c, "percent_scale", 100.0),
                out_name="pct_portafolio",
            )

        # Plotly figure
        x = ts.index
        y = ts["pct_portafolio"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers" if marker else "lines",
                name="% Impago portafolio",
                hovertemplate=(
                    "Mes: %{x|%Y-%m}"
                    "<br>% Impago: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

        # Mapear figsize -> pixeles aprox
        width = int(figsize[0] * 100)
        height = int(figsize[1] * 100)

        fig.update_layout(
            title=title,
            xaxis_title="Mes calendario",
            yaxis_title=f"% Impago medio de portafolio",
            template="plotly_white",
            hovermode="x", 
            width=width,
            height=height,
        )

        # grid visual similar a matplotlib
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        if show:
            fig.show()
            fig_out = None
        else:
            fig_out = fig

        if debug:
            return fig_out, ts

        return fig_out

