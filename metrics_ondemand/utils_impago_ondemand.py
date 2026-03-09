"""
utils_impago_ondemand.py

Motor “on-demand” para demo de Matriz de Impago (Datta) usando df_master_dummy.

Objetivo
--------
Flujo completo:
1) aplicar filtros on-demand (antes de agregación),
2) computar matriz vintage (cosecha x MOB) en modo exposure o ever,
3) graficar curva promedio por MOB,
4) graficar exposición por cosecha (sum Monto Fondeado),
5) graficar heatmap con annot únicamente en la “última diagonal” (último MOB observado por cosecha),
6) (opcional) guardar CSVs + PNGs por escenario en outputs/<scenario_name>/.

Requisitos / Suposiciones
-------------------------
- df_master_dummy ya contiene:
    - cohorte_col = "cosecha" (YYYY-MM)
    - mob_col = "MOB"
    - folio_col = "folio"
    - monto_col = "Monto Fondeado"
    - tipo_mora cols: BGI2+, BGI3+, BGI4+, BGI5+, CAST_VAL
    - ever cols: BGI2+_CONTEO_EVER, ..., BGI5+_CONTEO_EVER, CAST_CONTEO

- NO depende de "Fecha de Inicio". No llama ningún método que la requiera.

- Matplotlib:
    - 1 figura por plot
    - no fija colores explícitos (usa defaults)

Uso rápido
----------
from utils_impago_ondemand import ImpagoOnDemandEngine, Scenario

engine = ImpagoOnDemandEngine(df_master_dummy)
scenarios = [
    Scenario(name="S00_base_BGI4_exposure", tipo_mora="BGI4+", metric_mode="exposure", filters={}),
    Scenario(name="S01_CDMX_MTY_BGI4_exposure", tipo_mora="BGI4+", metric_mode="exposure",
             filters={"Ciudad Atria": ["CDMX", "Monterrey"]}),
    Scenario(name="S05_CAST_ever", tipo_mora="CAST_VAL", metric_mode="ever", filters={}),
]
engine.run_many(scenarios, show=True, save_outputs=False)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

TipoMora = Literal["BGI2+", "BGI3+", "BGI4+", "BGI5+", "CAST_VAL"]
MetricMode = Literal["exposure", "ever"]
EverDenMode = Literal["fixed", "variable"]
FilterDict = Dict[str, List[object]]


@dataclass(frozen=True)
class Scenario:
    """
    Escenario de ejecución on-demand para el cálculo y visualización
    de métricas de impago (exposure o EVER), bajo un conjunto de filtros
    y, opcionalmente, una desagregación por una sola dimensión.

    Un Scenario define *qué* se calcula y *cómo* se visualiza dentro del
    flujo estándar de `run_scenario`.

    Attributes
    ----------
    name:
        Nombre del escenario.
        Se utiliza para identificar el experimento y se sanitiza para uso
        en filesystem cuando se guardan outputs (CSVs e imágenes).

    tipo_mora:
        Identificador de la mora a evaluar.

        - En modo "exposure":
            Debe corresponder a una columna numérica de monto de mora,
            por ejemplo: "BGI2+", "BGI3+", "BGI4+", "BGI5+" o "CAST_VAL".

        - En modo "ever":
            Se usa únicamente como referencia semántica para mapear a la
            columna de conteo correspondiente:
            "*_CONTEO_EVER" o "CAST_CONTEO".
            Internamente, estos conteos se convierten a un flag binario
            (ever_flag) para el cálculo del % EVER.

    metric_mode:
        Tipo de métrica a calcular.

        - "exposure":
            %impago = sum(tipo_mora) / sum(Monto Fondeado) * 100

        - "ever":
            %ever = (#folios con EVER) / (#folios observados) * 100

        El modo seleccionado impacta tanto los cálculos como el tipo de
        visualizaciones generadas.

    filters:
        Diccionario de filtros a aplicar sobre el dataset base, antes de
        cualquier agregación o cálculo.

        Ejemplo:
            {
                "Ciudad Atria": ["CDMX", "Monterrey"],
                "Plazo": ["36", "48"]
            }

        Todos los filtros se aplican de forma conjunta (AND lógico).

    breakdown_col:
        Columna opcional para desagregar los resultados por una sola
        dimensión adicional.

        - Si se define, `run_scenario` generará una visualización adicional
          de curvas por MOB, una por cada valor relevante de esta dimensión,
          respetando los filtros del escenario.
        - Solo se permite una dimensión de desagregado por escenario.
        - Puede coincidir con una de las dimensiones usadas en `filters`.
          En ese caso, la desagregación producirá una o pocas curvas.

        Ejemplos válidos:
            "Perfil"
            "Plazo"
            "Cluster Mora"
            "Ciudad Atria"
    """
    name: str
    tipo_mora: TipoMora
    metric_mode: MetricMode
    filters: FilterDict
    breakdown_col: Optional[str] = None


class ImpagoOnDemandEngine:
    """
    Engine encapsulado para computar y visualizar la Matriz de Impago “on-demand”.

    Este engine NO cambia la estructura de df_master_dummy.
    Solo realiza casts temporales para ordenar cosechas (YYYY-MM -> datetime) para plots/sorts.

    Principios de diseño
    --------------------
    - Separación clara:
        apply_filters()       -> filtra filas
        compute_matrix()      -> produce (agg_long, matrix_wide)
        compute_curve_by_mob()-> produce serie por MOB
        compute_exposure_by_cosecha() -> exposición por cohorte
        plot_*()              -> visualizaciones (1 figura por función)
        run_scenario()        -> pipeline completo para un escenario
    - Robustez:
        valida columnas requeridas y evita división entre 0 (NaN).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        cohorte_col: str = "cosecha",
        mob_col: str = "MOB",
        folio_col: str = "folio",
        monto_col: str = "Monto Fondeado",
        ever_den_mode: EverDenMode = "variable",
        strict_filters: bool = True,
        outputs_base_dir: str = "output_data",
        debug: bool = False,
        castigo_enabled: bool = False,
        castigo_col: str = "Estatus Legal",
        castigo_exclude_values: Optional[List[str]] = None,
        castigo_strict: Optional[bool] = None,
    ):
        self.df = df
        self.cohorte_col = cohorte_col
        self.mob_col = mob_col
        self.folio_col = folio_col
        self.monto_col = monto_col
        self.ever_den_mode = ever_den_mode
        self.strict_filters = strict_filters
        self.outputs_base_dir = outputs_base_dir
        self.debug = debug
        self.castigo_enabled = castigo_enabled
        self.castigo_col = castigo_col
        self.castigo_exclude_values = castigo_exclude_values or ["Fraude", "Otros"]
        self.castigo_strict = self.strict_filters if castigo_strict is None else castigo_strict

        self._validate_min_contract()

    # Para debug
    def _log(self, msg: str, *, debug: Optional[bool] = None) -> None:
        """
        Logger simple controlado por bandera debug.
        - debug=None -> usa self.debug
        - debug=True/False -> override para esta llamada
        """
        flag = self.debug if debug is None else debug
        if flag:
            print(msg)


    # -----------------------------
    # Contract / validation
    # -----------------------------
    def _validate_min_contract(self) -> None:
        required = {self.folio_col, self.mob_col, self.cohorte_col, self.monto_col}
        missing = sorted(required - set(self.df.columns))
        if missing:
            raise KeyError(f"df no cumple contrato mínimo. Faltan columnas: {missing}")

    @staticmethod
    def ever_col_from_tipo_mora(tipo_mora: TipoMora) -> str:
        """Mapeo de tipo_mora a columna ever correspondiente."""
        if tipo_mora == "CAST_VAL":
            return "CAST_CONTEO"
        return f"{tipo_mora}_CONTEO_EVER"


    # -----------------------------
    # Filters
    # -----------------------------
    def apply_filters(self, df: pd.DataFrame, filters: Optional[FilterDict]) -> pd.DataFrame:
        """
        Aplica filtros tipo: {"Ciudad Atria": ["CDMX", "Monterrey"]}

        strict_filters:
            True  -> si columna no existe: error
            False -> ignora filtro inválido
        """
        if not filters:
            return df

        out = df
        for col, allowed in filters.items():
            if col not in out.columns:
                if self.strict_filters:
                    raise KeyError(f"Filtro inválido: columna '{col}' no existe en df.")
                continue

            s = out[col].astype(str).str.strip()
            allowed_norm = [str(x).strip() for x in allowed]

            out = out.loc[s.isin(allowed_norm)]
        return out

    def apply_castigo_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtro global (switch) para excluir castigos por Estatus Legal.
        Si castigo_enabled=False, regresa df sin cambios.
        """
        if not getattr(self, "castigo_enabled", False):
            return df

        col = getattr(self, "castigo_col", "Estatus Legal")
        exclude_vals = getattr(self, "castigo_exclude_values", ["Fraude", "Otros"])
        strict = getattr(self, "castigo_strict", True)

        if col not in df.columns:
            if strict:
                raise KeyError(f"Castigo filter activo pero columna '{col}' no existe en df.")
            self._log(f"[castigo] WARNING: columna '{col}' no existe. Se omite castigo filter.", debug=True)
            return df

        # Normalización tipo apply_filters: str + strip
        s = df[col].astype(str).str.strip()
        exclude_norm = [str(x).strip() for x in exclude_vals]

        out = df.loc[~s.isin(exclude_norm)]
        return out


    # -----------------------------
    # Core computations
    # -----------------------------
    def compute_matrix(
        self,
        df_filtered: pd.DataFrame,
        *,
        tipo_mora: TipoMora,
        metric_mode: MetricMode,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computa:
        - agg_long: granularidad (cosecha, MOB) con numeradores/denominadores agregados
        - matrix_wide: pivot cosecha x MOB con pct final

        Definiciones:
        - exposure:
            pct = sum(tipo_mora) / sum(Monto Fondeado) * 100
        - ever (solo soporte variable):
            pct = sum(ever_flag) / folio_den * 100
            donde:
            - ever_flag = 1 si ever_col > 0, else 0   (IMPORTANTE: convierte *_CONTEO_EVER a binario)
            - folio_den = #folios observados en (cosecha, MOB)  (nunique folio)

        Nota:
        - En esta versión SOLO se soporta ever_den_mode == "variable" para evitar inconsistencias.
        """

        # -------------------------
        # Validaciones mínimas
        # -------------------------
        if self.cohorte_col not in df_filtered.columns:
            raise KeyError(f"Columna cohorte '{self.cohorte_col}' no existe en df_filtered.")
        if self.mob_col not in df_filtered.columns:
            raise KeyError(f"Columna MOB '{self.mob_col}' no existe en df_filtered.")
        if self.folio_col not in df_filtered.columns:
            raise KeyError(f"Columna folio '{self.folio_col}' no existe en df_filtered.")

        if metric_mode == "exposure":
            if tipo_mora not in df_filtered.columns:
                raise KeyError(f"tipo_mora '{tipo_mora}' no existe como columna en df_filtered.")
            if self.monto_col not in df_filtered.columns:
                raise KeyError(f"Columna monto '{self.monto_col}' no existe en df_filtered.")
        else:
            ever_col = self.ever_col_from_tipo_mora(tipo_mora)
            if ever_col not in df_filtered.columns:
                raise KeyError(
                    f"Columna ever '{ever_col}' no existe en df_filtered (tipo_mora={tipo_mora})."
                )
            if getattr(self, "ever_den_mode", "variable") != "variable":
                raise ValueError(
                    "compute_matrix (ever) en este demo solo soporta ever_den_mode='variable'."
                )

        # Base group keys
        keys = [self.cohorte_col, self.mob_col]

        # -------------------------
        # Agregaciones base
        # -------------------------
        if metric_mode == "exposure":
            agg = (
                df_filtered
                .groupby(keys, as_index=False)
                .agg(
                    bgi_sum=(tipo_mora, "sum"),
                    monto_sum=(self.monto_col, "sum"),
                )
            )
            den = agg["monto_sum"].astype(float)
            num = agg["bgi_sum"].astype(float)
            agg["pct"] = np.where(den > 0, (num / den) * 100.0, np.nan)

        else:
            # EVER (fixed) usando ever_flag binario derivado de *_CONTEO_EVER
            ever_col = self.ever_col_from_tipo_mora(tipo_mora)

            # Copia mínima para no mutar df_filtered
            tmp = df_filtered[[self.cohorte_col, self.mob_col, self.folio_col, ever_col]].copy()

            # Convertir a flag binario (si es conteo/acumulado, >0 => 1)
            tmp["_ever_flag"] = (
                pd.to_numeric(tmp[ever_col], errors="coerce")
                .fillna(0)
                .gt(0)
                .astype(int)
            )

            # Denominador FIXED: #folios por cosecha (constante para todos los MOB)
            den_fixed = (
                tmp.groupby(self.cohorte_col, as_index=False)
                .agg(folio_den=(self.folio_col, "nunique"))
            )

            # Numerador por (cosecha, MOB): #folios con ever (sum de flag)
            agg = (
                tmp.groupby(keys, as_index=False)
                .agg(
                    ever_sum=("_ever_flag", "sum"),
                )
                .merge(den_fixed, on=self.cohorte_col, how="left")
            )

            den = agg["folio_den"].astype(float)
            num = agg["ever_sum"].astype(float)
            agg["pct"] = np.where(den > 0, (num / den) * 100.0, np.nan)


        # -------------------------
        # Pivot a matriz
        # -------------------------
        mat = agg.pivot(index=self.cohorte_col, columns=self.mob_col, values="pct")

        # Orden cronológico de cosecha sin mutar df: casteo local para sort del index
        idx_dt = pd.to_datetime(mat.index, errors="coerce")
        mat = mat.set_index(idx_dt)
        mat.index.name = self.cohorte_col
        mat = mat.sort_index()

        # Orden numérico de MOB si aplica
        try:
            mat = mat.reindex(sorted(mat.columns), axis=1)
        except Exception:
            pass

        return agg, mat

    def compute_curve_by_mob(
        self,
        agg_long: pd.DataFrame,
        *,
        metric_mode: MetricMode,
        decimals: int = 1, 
    ) -> pd.DataFrame:
        """
        Curva agregada por MOB.

        exposure:
            pct_impago_mob = sum(bgi_sum)/sum(monto_sum)*100
        ever:
            pct_ever_mob   = sum(ever_sum)/sum(folio_den)*100

        decimals:
            Número de decimales para redondeo del porcentaje (default = 1)
        """
        if metric_mode == "exposure":
            needed = {"bgi_sum", "monto_sum", self.mob_col}
            if not needed.issubset(set(agg_long.columns)):
                raise KeyError(f"agg_long no trae columnas para exposure: {sorted(needed)}")

            g = (
                agg_long.groupby(self.mob_col, as_index=False)
                .agg(bgi_sum=("bgi_sum", "sum"), monto_sum=("monto_sum", "sum"))
            )
            den = g["monto_sum"].astype(float)
            num = g["bgi_sum"].astype(float)

            g["pct_impago_mob"] = np.where(
                den > 0,
                (num / den) * 100.0,
                np.nan,
            ).round(decimals)

            return g.sort_values(self.mob_col)

        # --- EVER
        needed = {"ever_sum", "folio_den", self.mob_col}
        if not needed.issubset(set(agg_long.columns)):
            raise KeyError(f"agg_long no trae columnas para ever: {sorted(needed)}")

        g = (
            agg_long.groupby(self.mob_col, as_index=False)
            .agg(ever_sum=("ever_sum", "sum"), folio_den=("folio_den", "sum"))
        )
        den = g["folio_den"].astype(float)
        num = g["ever_sum"].astype(float)

        g["pct_ever_mob"] = np.where(
            den > 0,
            (num / den) * 100.0,
            np.nan,
        ).round(decimals)

        return g.sort_values(self.mob_col)

    def compute_exposure_by_cosecha(self, df_filtered: pd.DataFrame) -> pd.DataFrame:
        """
        Monto Fondeado (formato cliente):
        exposure = sum(Monto Fondeado) SOLO para registros con MOB == 1,
        después de aplicar filtros.

        Esto aproxima el "monto originado" de la cohorte.
        """
        if self.mob_col not in df_filtered.columns:
            raise KeyError(f"No existe columna MOB '{self.mob_col}' en df_filtered.")
        if self.monto_col not in df_filtered.columns:
            raise KeyError(f"No existe columna monto '{self.monto_col}' en df_filtered.")
        if self.cohorte_col not in df_filtered.columns:
            raise KeyError(f"No existe columna cohorte '{self.cohorte_col}' en df_filtered.")

        df_mob1 = df_filtered.loc[df_filtered[self.mob_col] == 1].copy()

        g = (
            df_mob1.groupby(self.cohorte_col, as_index=False)
            .agg(exposure=(self.monto_col, "sum"))
        )
        g["_cosecha_dt"] = pd.to_datetime(g[self.cohorte_col], errors="coerce")
        g = g.sort_values("_cosecha_dt").drop(columns=["_cosecha_dt"])

        return g


    # -----------------------------
    # Plotters
    # -----------------------------
    @staticmethod
    def plot_curve_by_mob(curve_df: pd.DataFrame, *, mob_col: str, y_col: str, title: str, show: bool = True):
        ATRIA_PURPLE = "#7A3EB1"
        plt.figure(figsize=(10, 5))
        plt.plot(curve_df[mob_col], curve_df[y_col], marker="o", color=ATRIA_PURPLE)
        plt.xlabel("MOB")
        plt.ylabel("% Impago")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        if show:
            plt.show()

    @staticmethod
    def plot_exposure_by_cosecha(expo_df: pd.DataFrame, *, x_col: str, y_col: str, title: str, show: bool = True):
        ATRIA_PURPLE = "#7A3EB1"
        x_dt = pd.to_datetime(expo_df[x_col], errors="coerce")
        plt.figure(figsize=(10, 5))
        plt.plot(x_dt, expo_df[y_col], marker="o")
        plt.xlabel("Cosecha")
        plt.ylabel("Monto Fondeado")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        if show:
            plt.show()

    @staticmethod
    def plot_heatmap_last_diagonal_annot(
        matrix_dt_index: pd.DataFrame,
        *,
        title: str,
        show: bool = True,
        fmt: str = "{:.2f}",
        fontsize: int = 9,
    ):
        """
        Heatmap de matriz con annot en “última diagonal”:
        - Por cada fila (cosecha), ubica el último MOB con valor no-NaN y lo anota.
        """
        m = matrix_dt_index.copy()

        # Asegurar orden
        m = m.sort_index()
        try:
            m = m.reindex(sorted(m.columns), axis=1)
        except Exception:
            pass

        plt.figure(figsize=(12, 6))
        im = plt.imshow(m.values, cmap="Purples")
        plt.colorbar(im, label="%")

        plt.title(title)
        plt.xlabel("MOB")
        plt.ylabel("Cosecha")

        # ticks
        plt.xticks(range(len(m.columns)), m.columns)
        plt.yticks(
            range(len(m.index)),
            [d.strftime("%Y-%m") if pd.notna(d) else "NaT" for d in m.index]
        )

        # annot diagonal (último valor observado por fila)
        arr = m.values
        vals = arr[~np.isnan(arr)]
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
            color = semaforo_color(val)
            plt.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
                # path_effects=[pe.withStroke(linewidth=2.5, foreground="black")]
            )

        plt.tight_layout()
        if show:
            plt.show()

    def plot_curve_by_mob_breakdown(
        self,
        scenario: Scenario,
        *,
        breakdown_col: str = "Perfil",
        max_levels: int = 8,
        min_folios: int = 200,
        show: bool = True,
        debug: Optional[bool] = None,
    ) -> plt.Figure:
        """
        (2) Curvas desagregadas por MOB, respetando filtros del escenario.

        - Aplica scenario.filters al df base.
        - Desagrega por breakdown_col (una sola dimensión).
        - Para cada nivel (valor de breakdown_col) calcula:
            agg_long_sub, _ = compute_matrix(...)
            curve_sub       = compute_curve_by_mob(agg_long_sub,...)
        y lo plotea en la misma figura.

        Orden de la leyenda:
        - Se ordena de menor a mayor según el valor final (último MOB disponible) de la métrica.
        Esto hace el plot más legible para demo/negocio.

        Parámetros
        ----------
        max_levels:
            Máximo número de niveles a graficar (para mantener legibilidad).
        min_folios:
            Mínimo de folios únicos requeridos por nivel para graficarlo (evita ruido).
        """
        if breakdown_col not in self.df.columns:
            raise KeyError(f"breakdown_col '{breakdown_col}' no existe en df.")

        # Base filtrada por escenario
        df_base = self.apply_filters(self.df, scenario.filters).copy()
        if df_base.empty:
            raise ValueError(
                f"Escenario '{scenario.name}' dejó 0 filas tras filtros: {scenario.filters}."
            )
        
        # Castigo global (switch) también en breakdown
        df_base = self.apply_castigo_filter(df_base).copy()
        if df_base.empty:
            raise ValueError(
                f"Escenario '{scenario.name}' quedó vacío tras castigo_enabled={self.castigo_enabled} "
                f"({self.castigo_col} excluye {self.castigo_exclude_values})."
            )

        # Niveles candidatos
        levels = (
            pd.Series(df_base[breakdown_col].astype(str).str.strip().unique())
            .dropna()
            .tolist()
        )

        # Soporte por nivel (#folios) para filtrar/ordenar
        level_stats = []
        for lv in levels:
            df_lv = df_base.loc[df_base[breakdown_col].astype(str).str.strip().eq(lv)]
            n_folios = df_lv[self.folio_col].nunique()
            level_stats.append((lv, n_folios))

        # Ordena por soporte desc, filtra por mínimo y recorta a max_levels
        level_stats.sort(key=lambda t: t[1], reverse=False)
        level_stats = [(lv, n) for (lv, n) in level_stats if n >= min_folios]
        level_stats = level_stats[:max_levels]

        if not level_stats:
            raise ValueError(
                f"No hay niveles en '{breakdown_col}' con >= {min_folios} folios bajo filtros={scenario.filters}."
            )

        levels_final = [lv for (lv, _) in level_stats]
        self._log(
            f"[breakdown] {scenario.name} col={breakdown_col} levels={levels_final}",
            debug=debug
        )

        # Figura (sin fijar colores explícitos)
        fig = plt.figure(figsize=(10, 5))

        # Columna y según modo
        y_col = "pct_impago_mob" if scenario.metric_mode == "exposure" else "pct_ever_mob"

        # 1) Primero computar curvas y métrica final para poder ordenar la leyenda
        curves = []
        for lv, n_f in level_stats:
            df_lv = df_base.loc[df_base[breakdown_col].astype(str).str.strip().eq(lv)]
            if df_lv.empty:
                continue

            agg_sub, _ = self.compute_matrix(
                df_lv,
                tipo_mora=scenario.tipo_mora,
                metric_mode=scenario.metric_mode,
            )

            curve_sub = self.compute_curve_by_mob(
                agg_sub,
                metric_mode=scenario.metric_mode,
            )

            # Último valor válido para ordenar (menor -> mayor)
            s = curve_sub[y_col].dropna()
            last_val = s.iloc[-1] if not s.empty else np.nan

            curves.append(
                {
                    "label": f"{lv} (n={n_f})",
                    "curve": curve_sub,
                    "last_val": float(last_val) if pd.notna(last_val) else np.nan,
                }
            )

        # Ordena por last_val asc, NaNs al final
        curves = sorted(curves, key=lambda d: (np.isnan(d["last_val"]), d["last_val"]))

        # 2) Plotear ya ordenado para que la legend quede ordenada
        for item in curves:
            c = item["curve"]
            plt.plot(
                c[self.mob_col],
                c[y_col],
                marker="o",
                linewidth=1.5,
                label=item["label"],
            )

        plt.xlabel("MOB")
        plt.ylabel("% Impago" if scenario.metric_mode == "exposure" else "% EVER")
        plt.title(f"Curvas por MOB (desagregado por {breakdown_col}) – {scenario.name}")
        plt.grid(True)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()

        if show:
            plt.show()

        return fig



    # -----------------------------
    # IO helpers
    # -----------------------------
    @staticmethod
    def sanitize_name(name: str) -> str:
        """
        Convierte scenario_name a algo seguro para filesystem.
        """
        name = name.strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^A-Za-z0-9_\-\.]+", "", name)
        return name or "scenario"

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def save_outputs(
        self,
        scenario_name: str,
        *,
        agg_long: pd.DataFrame,
        matrix_dt_index: pd.DataFrame,
        curve_by_mob: pd.DataFrame,
        exposure_by_cosecha: pd.DataFrame,
        figs: Dict[str, plt.Figure],
    ) -> str:
        """
        Guarda:
        - matrix.csv, curve_mob.csv, exposure_by_cosecha.csv
        - matrix.png (tabla NO se guarda como imagen aquí), curve_mob.png, exposure_by_cosecha.png, heatmap.png
        Retorna el path del directorio del escenario.
        """
        safe = self.sanitize_name(scenario_name)
        out_dir = os.path.join(self.outputs_base_dir, safe)
        self._ensure_dir(out_dir)

        # CSVs
        agg_long.to_csv(os.path.join(out_dir, "agg_long.csv"), index=False)

        # matrix: convertir index datetime -> YYYY-MM
        mat_to_save = matrix_dt_index.copy()
        def _fmt_idx(d):
            if pd.isna(d):
                return ""
            # datetime-like
            if hasattr(d, "strftime"):
                return d.strftime("%Y-%m")
            # string / other
            return str(d)

        mat_to_save.index = [_fmt_idx(d) for d in mat_to_save.index]
        mat_to_save.to_csv(os.path.join(out_dir, "matrix.csv"))

        curve_by_mob.to_csv(os.path.join(out_dir, "curve_mob.csv"), index=False)
        exposure_by_cosecha.to_csv(os.path.join(out_dir, "exposure_by_cosecha.csv"), index=False)

        # PNGs
        for key, fig in figs.items():
            fig.savefig(os.path.join(out_dir, f"{key}.png"), dpi=160, bbox_inches="tight")

        return out_dir

    def make_matrix_display(
        self,
        matrix_dt: pd.DataFrame,
        *,
        cell_decimals: int = 2,
        summary_decimals: int = 1,
        blank: str = "",
        add_blank_row: bool = True,
        add_mob_summary_row: bool = True,
        summary_label: str = "2+",
        agg_long: Optional[pd.DataFrame] = None,
        metric_mode: Optional[MetricMode] = None,
        exposure_by_cosecha: Optional[pd.DataFrame] = None,  
        exposure_col_label: str = "$",                       
    ) -> pd.DataFrame:
        """
        Matriz display:
        - filas: cosechas (YYYY-MM)
        - primera columna (opcional): $ exposure (MOB==1)
        - celdas: % con cell_decimals
        - fila resumen por MOB: % con summary_decimals
        """
        m_num = matrix_dt.copy()

        # --- index YYYY-MM
        idx_str = [d.strftime("%Y-%m") if pd.notna(d) else blank for d in m_num.index]
        m_num.index = idx_str

        def _fmt_cell(x):
            if pd.isna(x):
                return blank
            return f"{float(x):.{cell_decimals}f}%"

        def _fmt_summary(x):
            if pd.isna(x):
                return blank
            return f"{float(x):.{summary_decimals}f}%"

        # Matriz principal (2 decimales)
        m_disp = m_num.map(_fmt_cell)

        # --- insertar columna $ si viene exposure_by_cosecha
        if exposure_by_cosecha is not None:
            if self.cohorte_col not in exposure_by_cosecha.columns or "exposure" not in exposure_by_cosecha.columns:
                raise KeyError("exposure_by_cosecha debe traer columnas: [cosecha, exposure].")

            expo = exposure_by_cosecha.copy()
            expo["_cosecha_dt"] = pd.to_datetime(expo[self.cohorte_col], errors="coerce")
            expo = expo.sort_values("_cosecha_dt").drop(columns=["_cosecha_dt"])

            expo_idx = [pd.to_datetime(x, errors="coerce").strftime("%Y-%m") if pd.notna(pd.to_datetime(x, errors="coerce")) else blank
                        for x in expo[self.cohorte_col]]

            expo_series = pd.Series(expo["exposure"].values, index=expo_idx)

            def _fmt_money(v):
                if pd.isna(v):
                    return blank
                try:
                    return f"{float(v):,.0f}"
                except Exception:
                    return blank

            m_disp.insert(0, exposure_col_label, expo_series.reindex(m_disp.index).map(_fmt_money))

        # fila en blanco 
        if add_blank_row:
            empty_row = pd.DataFrame([[blank] * m_disp.shape[1]], columns=m_disp.columns, index=[blank])
            m_disp = pd.concat([m_disp, empty_row], axis=0)

        # fila resumen por MOB (1 decimal) usando compute_curve_by_mob
        if add_mob_summary_row:
            if agg_long is None or metric_mode is None:
                raise ValueError("add_mob_summary_row=True requiere agg_long y metric_mode.")

            curve_df = self.compute_curve_by_mob(agg_long, metric_mode=metric_mode, decimals=summary_decimals)
            y_col = "pct_impago_mob" if metric_mode == "exposure" else "pct_ever_mob"
            summary_series = curve_df.set_index(self.mob_col)[y_col]

            # si hay columna $, dejamos en blanco
            if exposure_col_label in m_disp.columns:
                summary_series = summary_series.reindex([c for c in m_disp.columns if c != exposure_col_label])
                summary_row = pd.DataFrame(
                    [[blank] + list(summary_series.reindex(m_disp.columns[1:]).values)],
                    columns=m_disp.columns,
                    index=[summary_label],
                )
            else:
                summary_row = pd.DataFrame([summary_series.reindex(m_disp.columns).values], columns=m_disp.columns, index=[summary_label])

            # formatear solo la parte de MOBs
            if exposure_col_label in summary_row.columns:
                mob_cols = [c for c in summary_row.columns if c != exposure_col_label]
                summary_row[mob_cols] = summary_row[mob_cols].map(_fmt_summary)
            else:
                summary_row = summary_row.map(_fmt_summary)

            m_disp = pd.concat([m_disp, summary_row], axis=0)

        return m_disp

    @staticmethod
    def summary_label_from_tipo_mora(tipo_mora: str) -> str:
        """
        Convierte tipo_mora a etiqueta de fila resumen:
        - "BGI4+" -> "4+"
        - "BGI3+" -> "3+"
        - "BGI2+" -> "2+"
        - "BGI5+" -> "5+"
        - "CAST_VAL" -> "CAST"
        """
        if tipo_mora == "CAST_VAL":
            return "CAST"
        if isinstance(tipo_mora, str) and tipo_mora.startswith("BGI") and tipo_mora.endswith("+"):
            # "BGI4+" -> "4+"
            return tipo_mora.replace("BGI", "")
        return str(tipo_mora)


    # -----------------------------
    # Scenario runners
    # -----------------------------
    def run_scenario(
        self,
        scenario: Scenario,
        *,
        show: bool = True,
        save_outputs: bool = False,
        debug: Optional[bool] = None,
    ) -> Dict[str, object]:
        """
        Ejecuta un escenario end-to-end.

        Retorna un dict con:
            - df_filtered
            - agg_long
            - matrix_dt
            - curve_mob
            - exposure_by_cosecha
            - out_dir (si save_outputs=True)
        """
        ATRIA_PURPLE = "#7A3EB1"
        df_f = self.apply_filters(self.df, scenario.filters).copy()
        if df_f.empty:
            raise ValueError(
                f"Escenario '{scenario.name}' dejó 0 filas tras filtros: {scenario.filters}. "
                "Revisa valores exactos o normalización."
            )
        
        # Castigo global (switch)
        rows_before = len(df_f)
        folios_before = df_f[self.folio_col].nunique() if self.folio_col in df_f.columns else None

        df_f = self.apply_castigo_filter(df_f).copy()

        rows_after = len(df_f)
        folios_after = df_f[self.folio_col].nunique() if self.folio_col in df_f.columns else None

        if df_f.empty:
            raise ValueError(
                f"Escenario '{scenario.name}' quedó vacío tras aplicar castigo_enabled={self.castigo_enabled} "
                f"({self.castigo_col} excluye {self.castigo_exclude_values})."
            )
        # print("rows:", len(df_f))
        # print("folios_unique:", df_f["folio"].nunique())
        # print("cosechas_unique:", df_f["cosecha"].nunique())
        # print("MOB_range:", df_f["MOB"].min(), df_f["MOB"].max())

        # revisa valores del filtro (para ver si matchean)
        # for k, v in scenario.filters.items():
        #     if k in df_f.columns:
        #         print(k, "unique (sample):", pd.Series(df_f[k].unique()).head(10).tolist())

        # Orden interno por cosecha (sin tocar contrato): ayuda a consistencia en logs
        df_f["_cosecha_dt"] = pd.to_datetime(df_f[self.cohorte_col], errors="coerce")
        df_f = df_f.sort_values(["_cosecha_dt", self.mob_col]).drop(columns=["_cosecha_dt"])

        agg_long, matrix_dt = self.compute_matrix(
            df_f,
            tipo_mora=scenario.tipo_mora,
            metric_mode=scenario.metric_mode,
        )

        curve = self.compute_curve_by_mob(agg_long, metric_mode=scenario.metric_mode)
        expo = self.compute_exposure_by_cosecha(df_f)

        # Logs breves
        mob_min = df_f[self.mob_col].min() if len(df_f) else None
        mob_max = df_f[self.mob_col].max() if len(df_f) else None
        self._log("\n" + "=" * 80, debug=debug)
        self._log(
            f"[{scenario.name}] tipo_mora={scenario.tipo_mora} "
            f"metric_mode={scenario.metric_mode}",
            debug=debug
        )
        self._log(f"filters={scenario.filters}", debug=debug)
        self._log(
            f"rows={len(df_f):,} | "
            f"folios_unique={df_f[self.folio_col].nunique():,} | "
            f"MOB={mob_min}..{mob_max}",
            debug=debug
        )

        # Plots (1 figura por plot)
        figs: Dict[str, plt.Figure] = {}

        # (2.1) Curva por MOB
        y_col = "pct_impago_mob" if scenario.metric_mode == "exposure" else "pct_ever_mob"
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(curve[self.mob_col], curve[y_col], marker="o", color=ATRIA_PURPLE)
        plt.xlabel("MOB")
        plt.ylabel("% Impago")
        plt.title(f"% Impago promedio por MOB – {scenario.name}")
        plt.grid(True)
        plt.tight_layout()
        figs["curve_mob"] = fig1
        if show:
            plt.show()

        # (2.2) Curvas desagregadas por MOB
        if getattr(scenario, "breakdown_col", None):
            fig_break = self.plot_curve_by_mob_breakdown(
                scenario,
                breakdown_col=scenario.breakdown_col,
                max_levels=10,
                min_folios=50,
                show=show,
                debug=debug,
            )
            figs["curve_mob_breakdown"] = fig_break

        # (3) Monto Fondeado
        fig2 = plt.figure(figsize=(10, 5))
        x_dt = pd.to_datetime(expo[self.cohorte_col], errors="coerce")
        plt.plot(x_dt, expo["exposure"], marker="o", color=ATRIA_PURPLE)
        plt.xlabel("Cosecha")
        plt.ylabel("Monto Fondeado")
        plt.title(f"Monto Fondeado – {scenario.name}")
        plt.grid(True)
        plt.tight_layout()
        figs["exposure_by_cosecha"] = fig2
        if show:
            plt.show()

        # (4) Heatmap con última diagonal annot
        # Creamos figura sin fijar colores explícitos.
        fig3 = plt.figure(figsize=(12, 6))
        m = matrix_dt.sort_index()
        try:
            m = m.reindex(sorted(m.columns), axis=1)
        except Exception:
            pass

        im = plt.imshow(m.values, aspect="auto", cmap="Purples")
        plt.colorbar(im, label="%")
        plt.title(f"% de Impago – {scenario.name}")
        plt.xlabel("MOB")
        plt.ylabel("Cosecha")

        plt.xticks(range(len(m.columns)), m.columns)
        plt.yticks(
            range(len(m.index)),
            [d.strftime("%Y-%m") if pd.notna(d) else "NaT" for d in m.index]
        )

        arr = m.values
        vals = arr[~np.isnan(arr)]
        p50, p80, p95 = np.percentile(vals, [50, 80, 95])

        def semaforo_color(x: float) -> str:
            if x >= p95:
                return "red"
            if x >= p80:
                return "orange"
            if x >= p50:
                return "green"
            return "#1b5e20"   # en celdas claras se ve; y con halo se ve en todas

        for i in range(arr.shape[0]):
            row = arr[i, :]
            valid = np.where(~np.isnan(row))[0]
            if len(valid) == 0:
                continue
            j = valid[-1]
            val = row[j]
            color = semaforo_color(val)
            plt.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
                # path_effects=[pe.withStroke(linewidth=2.5, foreground="black")]
            )

        plt.tight_layout()
        figs["heatmap"] = fig3
        if show:
            plt.show()

        summary_label = self.summary_label_from_tipo_mora(scenario.tipo_mora)

        matrix_display = self.make_matrix_display(
            matrix_dt,
            cell_decimals=2,
            summary_decimals=1,
            blank="",
            add_blank_row=True,
            add_mob_summary_row=True,
            summary_label=summary_label,  
            agg_long=agg_long,
            metric_mode=scenario.metric_mode,
            exposure_by_cosecha=expo,      
            exposure_col_label="$",
        )

        # (1) Matriz “como tabla”
        result: Dict[str, object] = {
            "df_filtered": df_f,
            "agg_long": agg_long,
            "matrix_dt": matrix_dt,
            "matrix_display": matrix_display,
            "curve_mob": curve,
            "exposure_by_cosecha": expo,

            # Metadata para Streamlit / debug
            "castigo_enabled": self.castigo_enabled,
            "castigo_col": self.castigo_col,
            "castigo_exclude_values": list(self.castigo_exclude_values),
            "rows_before_castigo": rows_before,
            "rows_after_castigo": rows_after,
            "folios_before_castigo": folios_before,
            "folios_after_castigo": folios_after,
        }

        if save_outputs:
            out_dir = self.save_outputs(
                scenario.name,
                agg_long=agg_long,
                matrix_dt_index=matrix_display,
                curve_by_mob=curve,
                exposure_by_cosecha=expo,
                figs=figs,
            )
            result["out_dir"] = out_dir
            self._log(f"Outputs guardados en: {out_dir}", debug=debug)

        return result

    def run_many(
        self,
        scenarios: List[Scenario],
        *,
        show: bool = True,
        save_outputs: bool = False,
    ) -> List[Dict[str, object]]:
        """
        Ejecuta múltiples escenarios.
        """
        results = []
        for sc in scenarios:
            results.append(self.run_scenario(sc, show=show, save_outputs=save_outputs))
        print("\nListo. Si esto ya responde bien a filtros, el siguiente paso es portar el runner a Streamlit.")
        return results
