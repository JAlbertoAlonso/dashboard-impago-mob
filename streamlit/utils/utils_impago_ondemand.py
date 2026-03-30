"""
utils_impago_ondemand.py

Motor “on-demand” para demo de Matriz de Impago (Datta) usando df_master_dummy.

Objetivo
--------
Flujo completo:
1) aplicar filtros on-demand (antes de agregación),
2) computar matriz vintage (cosecha x MOB) en modo cosechas o ever,
3) graficar curva promedio por MOB,
4) graficar exposición por cosecha (sum Monto Fondeado),
5) graficar heatmap con annot únicamente en la “última diagonal” (último MOB observado por cosecha),
6) (opcional) guardar CSVs + PNGs por escenario en outputs/<scenario_name>/.

Requisitos / Suposiciones
-------------------------
- df_master_dummy ya contiene:
    - cohorte_col = "cosecha" (YY-MM)
    - mob_col = "MOB"
    - folio_col = "folio"
    - monto_col = "Monto Fondeado"
    - tipo_mora cols: BGI2+, BGI3+, BGI4+, BGI5+, CAST
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
    Scenario(name="S00_base_BGI4_cosechas", tipo_mora="BGI4+", metric_mode="cosechas", filters={}),
    Scenario(name="S01_CDMX_MTY_BGI4_cosechas", tipo_mora="BGI4+", metric_mode="cosechas",
             filters={"Ciudad Atria": ["CDMX", "Monterrey"]}),
    Scenario(name="S05_CAST_ever", tipo_mora="CAST", metric_mode="ever", filters={}),
]
engine.run_many(scenarios, show=True, save_outputs=False)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

TipoMora = Literal["BGI2+", "BGI3+", "BGI4+", "BGI5+", "CAST"]
MetricMode = Literal["cosechas", "ever"]
EverDenMode = Literal["fixed", "variable"]
FilterDict = Dict[str, List[object]]


@dataclass(frozen=True)
class Scenario:
    """
    Escenario de ejecución on-demand para el cálculo y visualización
    de métricas de impago (cosechas o EVER), bajo un conjunto de filtros
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

        - En modo "cosechas":
            Debe corresponder a una columna numérica de monto de mora,
            por ejemplo: "BGI2+", "BGI3+", "BGI4+", "BGI5+" o "CAST".

        - En modo "ever":
            Se usa únicamente como referencia semántica para mapear a la
            columna de conteo correspondiente:
            "*_CONTEO_EVER" o "CAST_CONTEO".
            Internamente, estos conteos se convierten a un flag binario
            (ever_flag) para el cálculo del % EVER.

    metric_mode:
        Tipo de métrica a calcular.

        - "cosechas":
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
        legend_order_map: Optional[dict[str, list[str]]] = None,
        legend_order_path: Optional[str | Path] = None,
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
        df = df.copy()

        if "CAST" not in df.columns and "CAST_VAL" in df.columns:
            df = df.rename(columns={"CAST_VAL": "CAST"})

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
        if legend_order_path is None:
            default_path = Path(__file__).resolve().parents[1] / "config" / "dimensions_order.yml"
        else:
            default_path = Path(legend_order_path).resolve()

        if legend_order_map is not None:
            self.legend_order_map = legend_order_map
        else:
            self.legend_order_map = self.load_legend_order_map(default_path)

        self._validate_min_contract()


    # -----------------------------
    # Para debug
    # -----------------------------
    def _log(self, msg: str, *, debug: Optional[bool] = None) -> None:
        """
        Logger simple controlado por bandera debug.
        - debug=None -> usa self.debug
        - debug=True/False -> override para esta llamada
        """
        flag = self.debug if debug is None else debug
        if flag:
            print(msg)


    # ========================================
    # Validación de datos
    # ========================================
    def _validate_min_contract(self) -> None:
        required = {self.folio_col, self.mob_col, self.cohorte_col, self.monto_col}
        missing = sorted(required - set(self.df.columns))
        if missing:
            raise KeyError(f"df no cumple contrato mínimo. Faltan columnas: {missing}")

    @staticmethod
    def ever_col_from_tipo_mora(tipo_mora: TipoMora) -> str:
        """Mapeo de tipo_mora a columna ever correspondiente."""
        if tipo_mora == "CAST":
            return "CAST_CONTEO"
        return f"{tipo_mora}_CONTEO_EVER"


    # ========================================
    # Aplicación de filtros de escenario
    # ========================================
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
        Lógica de fraude / estatus legal.

        Semántica vigente:
        - castigo_enabled = True  -> "Con fraude"  -> incluir todos los registros
        - castigo_enabled = False -> "Sin fraude"  -> excluir Estatus Legal en exclude_values
        """
        # Si "Con fraude" está activo, NO se filtra nada
        if getattr(self, "castigo_enabled", False):
            return df

        col = getattr(self, "castigo_col", "Estatus Legal")
        exclude_vals = getattr(self, "castigo_exclude_values", ["Fraude", "Otros"])
        strict = getattr(self, "castigo_strict", True)

        if col not in df.columns:
            if strict:
                raise KeyError(f"Castigo filter activo pero columna '{col}' no existe en df.")
            self._log(f"[castigo] WARNING: columna '{col}' no existe. Se omite castigo filter.", debug=True)
            return df

        s = df[col].astype(str).str.strip()
        exclude_norm = [str(x).strip() for x in exclude_vals]

        out = df.loc[~s.isin(exclude_norm)]
        return out

    def _valid_breakdown_mask(self, df: pd.DataFrame, breakdown_col: str) -> pd.Series:
        """
        True solo para filas con valor real utilizable en la dimensión de breakdown.
        Excluye nulos, vacíos y tokens basura frecuentes.
        """
        if breakdown_col not in df.columns:
            raise KeyError(f"breakdown_col '{breakdown_col}' no existe en df.")

        s = df[breakdown_col]
        s_str = s.astype(str).str.strip()

        invalid_tokens = {"", "nan", "none", "<na>", "null", "n/a"}

        mask = s.notna() & ~s_str.str.lower().isin(invalid_tokens)
        return mask

    def _drop_missing_breakdown_rows(self, df: pd.DataFrame, breakdown_col: str) -> pd.DataFrame:
        """
        Elimina filas sin valor útil en breakdown_col y normaliza la columna.
        """
        out = df.copy()
        mask = self._valid_breakdown_mask(out, breakdown_col)
        out = out.loc[mask].copy()
        out[breakdown_col] = out[breakdown_col].astype(str).str.strip()
        return out

    def _get_valid_breakdown_levels(self, df: pd.DataFrame, breakdown_col: str) -> list[str]:
        """
        Devuelve solo niveles válidos de breakdown, ya limpios y sin basura.
        """
        tmp = self._drop_missing_breakdown_rows(df, breakdown_col)
        return sorted(tmp[breakdown_col].dropna().astype(str).str.strip().unique().tolist())
    
    def _filter_base_by_cohort_range(
        self,
        df: pd.DataFrame,
        cohort_start: str,
        cohort_end: str,
    ) -> pd.DataFrame:
        """
        Filtra un dataframe por rango inclusivo de cohortes base mensuales.

        Parámetros
        ----------
        df : pd.DataFrame
            Debe contener la columna 'cosecha' en formato YYYY-MM o parseable.
        cohort_start : str
            Cohorte inicial (incluida), por ejemplo '2022-06'.
        cohort_end : str
            Cohorte final (incluida), por ejemplo '2022-12'.
        """
        if df.empty:
            return df.copy()

        out = df.copy()

        out["_cosecha_dt_"] = pd.to_datetime(out["cosecha"], errors="coerce")
        out = out.loc[out["_cosecha_dt_"].notna()].copy()

        if out.empty:
            return out

        start_dt = pd.to_datetime(cohort_start, errors="coerce")
        end_dt = pd.to_datetime(cohort_end, errors="coerce")

        if pd.isna(start_dt) or pd.isna(end_dt):
            return out.iloc[0:0].copy()

        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        out = out.loc[
            (out["_cosecha_dt_"] >= start_dt) &
            (out["_cosecha_dt_"] <= end_dt)
        ].copy()

        return out

    def _build_bucket_from_cosecha(
        self,
        df: pd.DataFrame,
        freq_mode: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Construye la columna bucket y el bucket_order a partir de la cosecha base.
        La selección temporal ya debió haberse hecho antes, sobre cosechas mensuales.
        """
        if df.empty:
            out = df.copy()
            out["bucket"] = pd.Series(dtype="object")
            return out, []

        out = df.copy()

        if "_cosecha_dt_" in out.columns:
            dt = pd.to_datetime(out["_cosecha_dt_"], errors="coerce")
        else:
            dt = pd.to_datetime(out["cosecha"], errors="coerce")

        out = out.loc[dt.notna()].copy()
        dt = pd.to_datetime(out["cosecha"], errors="coerce")

        if out.empty:
            out["bucket"] = pd.Series(dtype="object")
            return out, []

        if freq_mode == "Mensual":
            out["bucket"] = dt.dt.strftime("%Y-%m")

            bucket_order = (
                pd.DataFrame({
                    "dt": dt,
                    "bucket": out["bucket"]
                })
                .drop_duplicates()
                .sort_values("dt")["bucket"]
                .tolist()
            )
        else:
            q_num = dt.dt.quarter
            q_roman = q_num.map({1: "I", 2: "II", 3: "III", 4: "IV"})

            out["bucket"] = dt.dt.year.astype(str) + "_" + q_roman

            bucket_order = (
                pd.DataFrame({
                    "dt": dt,
                    "bucket": out["bucket"]
                })
                .groupby("bucket", as_index=False)["dt"]
                .min()
                .sort_values("dt")["bucket"]
                .tolist()
            )

        return out, bucket_order
    
    def get_originacion_base(self, scenario, breakdown_col):
        """
        Obtiene la base de originación (MOB=1) bajo el escenario filtrado.
        Se usará para las gráficas de composición por cosecha.
        """

        # aplicar filtros del escenario
        df_f = self.apply_filters(self.df, scenario.filters)

        # aplicar filtro de castigo/fraude según tipo de mora
        df_f = self.apply_castigo_filter(df_f)

        # originación = solo MOB 1
        df_f = df_f[df_f["MOB"] == 1].copy()

        # columnas necesarias
        cols = [
            "cosecha",
            "folio",
            "Monto Fondeado",
            breakdown_col
        ]

        df_f = df_f[cols]

        # conservar solo filas con breakdown disponible
        df_f = self._drop_missing_breakdown_rows(df_f, breakdown_col)

        return df_f
    
    def compute_breakdown_composition(
        self,
        scenario,
        breakdown_col,
        freq_mode,
        value_mode,
        cohort_start,
        cohort_end,
    ):
        """
        Calcula la composición por cohorte para la gráfica de barras apiladas.

        Regla temporal:
        - primero se filtran las cosechas base mensuales dentro del rango [cohort_start, cohort_end]
        - después se agregan a mensual o trimestral para visualización
        """

        df = self.get_originacion_base(scenario, breakdown_col)

        if df.empty:
            return pd.DataFrame(columns=["bucket", breakdown_col, "value", "total", "pct"]), []

        # -------------------------------------
        # Filtrar rango explícito de cohortes base
        # -------------------------------------
        df = self._filter_base_by_cohort_range(
            df=df,
            cohort_start=cohort_start,
            cohort_end=cohort_end,
        )

        if df.empty:
            return pd.DataFrame(columns=["bucket", breakdown_col, "value", "total", "pct"]), []

        # -------------------------
        # Construir bucket temporal
        # -------------------------
        df, bucket_order = self._build_bucket_from_cosecha(
            df=df,
            freq_mode=freq_mode,
        )

        if df.empty or not bucket_order:
            return pd.DataFrame(columns=["bucket", breakdown_col, "value", "total", "pct"]), []

        # -------------------------
        # Agregación
        # -------------------------
        if value_mode == "Monto fondeado":
            agg = (
                df.groupby(["bucket", breakdown_col])["Monto Fondeado"]
                .sum()
                .reset_index(name="value")
            )
        else:
            agg = (
                df.groupby(["bucket", breakdown_col])["folio"]
                .nunique()
                .reset_index(name="value")
            )

        if agg.empty:
            return pd.DataFrame(columns=["bucket", breakdown_col, "value", "total", "pct"]), []

        # -------------------------
        # Calcular porcentajes
        # -------------------------
        totals = (
            agg.groupby("bucket")["value"]
            .sum()
            .reset_index(name="total")
        )

        agg = agg.merge(totals, on="bucket", how="left")
        agg["pct"] = np.where(agg["total"] > 0, agg["value"] / agg["total"], np.nan)

        # -------------------------
        # Barra Total
        # -------------------------
        total = (
            agg.groupby(breakdown_col)["value"]
            .sum()
            .reset_index()
        )

        total["bucket"] = "Total"
        total["total"] = total["value"].sum()
        total["pct"] = np.where(total["total"] > 0, total["value"] / total["total"], np.nan)

        agg = pd.concat([agg, total], ignore_index=True)

        bucket_order = bucket_order + ["Total"]

        return agg, bucket_order
    
    def get_transversal_base(self, scenario, breakdown_col, mob_fix):
        """
        Base para la gráfica de tendencias transversales:
        universo filtrado por escenario y observado en un MOB fijo.

        Trae todo lo necesario para calcular:
        - contribución monetaria en modo 'cosechas'
        - contribución por conteo en modo 'ever'
        """

        # aplicar filtros del escenario
        df_f = self.apply_filters(self.df, scenario.filters)

        # aplicar lógica de castigo / fraude como en el resto del engine
        df_f = self.apply_castigo_filter(df_f)

        # quedarnos solo con el MOB fijo
        df_f = df_f[df_f[self.mob_col] == mob_fix].copy()

        # columnas mínimas necesarias
        cols = [
            self.cohorte_col,
            self.folio_col,
            breakdown_col,
        ]

        # en cosechas necesitamos la columna de mora y monto
        if scenario.metric_mode == "cosechas":
            cols.append(scenario.tipo_mora)
            if self.monto_col in df_f.columns:
                cols.append(self.monto_col)

        # en ever necesitamos la ever column real
        else:
            ever_col = self.ever_col_from_tipo_mora(scenario.tipo_mora)
            if ever_col in df_f.columns:
                cols.append(ever_col)

        cols = [c for c in cols if c in df_f.columns]
        df_f = df_f[cols].copy()

        # conservar solo filas con breakdown disponible
        df_f = self._drop_missing_breakdown_rows(df_f, breakdown_col)

        return df_f

    def compute_breakdown_transversal_trends(
        self,
        scenario,
        breakdown_col,
        mob_fix,
        freq_mode,
        cohort_start,
        cohort_end,
    ):
        """
        Calcula la tabla base para la gráfica de tendencias transversales
        como contribución de cada segmento al % total de impago del corte
        transversal (bucket + MOB fijo).

        Nueva semántica:
        - Ya no mide la tasa interna de cada segmento.
        - Ahora mide cuánto aporta cada segmento al % total del bucket,
        de forma consistente con la matriz en el MOB fijo.

        Reglas:
        - metric_mode == "cosechas":
            contrib_seg = sum(tipo_mora del segmento) / sum(Monto Fondeado total del bucket) * 100

        - metric_mode == "ever":
            contrib_seg = sum(ever_flag del segmento) / folios totales del bucket * 100

        Esto garantiza que:
        - la suma de contribuciones por segmento en cada bucket
        sea igual al valor total del corte transversal;
        - y que si solo vive un segmento por filtros,
        ese segmento coincida exactamente con la matriz.
        """

        df = self.get_transversal_base(
            scenario=scenario,
            breakdown_col=breakdown_col,
            mob_fix=mob_fix,
        ).copy()

        if df.empty:
            return pd.DataFrame(columns=["bucket", breakdown_col, "num_seg", "den_total", "pct"]), []

        # -------------------------------------
        # Filtrar rango explícito de cohortes base
        # -------------------------------------
        df = self._filter_base_by_cohort_range(
            df=df,
            cohort_start=cohort_start,
            cohort_end=cohort_end,
        )

        if df.empty:
            return pd.DataFrame(columns=["bucket", breakdown_col, "num_seg", "den_total", "pct"]), []

        # -------------------------
        # Construir bucket temporal
        # -------------------------
        df, bucket_order = self._build_bucket_from_cosecha(
            df=df,
            freq_mode=freq_mode,
        )

        if df.empty or not bucket_order:
            return pd.DataFrame(columns=["bucket", breakdown_col, "num_seg", "den_total", "pct"]), []

        # =========================================================
        # MODO COSECHAS -> contribución monetaria al % total
        # =========================================================
        if scenario.metric_mode == "cosechas":
            if scenario.tipo_mora not in df.columns:
                raise KeyError(
                    f"tipo_mora '{scenario.tipo_mora}' no existe en la base transversal."
                )
            if self.monto_col not in df.columns:
                raise KeyError(
                    f"La base transversal no contiene '{self.monto_col}' para cálculo en modo cosechas."
                )

            df_c = df.copy()

            # Numerador del segmento = suma de mora monetaria del segmento
            df_c["_mora_num_"] = pd.to_numeric(
                df_c[scenario.tipo_mora], errors="coerce"
            ).fillna(0.0)

            # Denominador total del bucket = suma total de monto fondeado del bucket
            df_c["_monto_den_"] = pd.to_numeric(
                df_c[self.monto_col], errors="coerce"
            ).fillna(0.0)

            # Numerador por segmento dentro del bucket
            agg_seg = (
                df_c.groupby(["bucket", breakdown_col], as_index=False)
                .agg(
                    num_seg=("_mora_num_", "sum"),
                )
            )

            # Denominador total del bucket (NO por segmento)
            den_total = (
                df_c.groupby("bucket", as_index=False)
                .agg(
                    den_total=("_monto_den_", "sum"),
                )
            )

            agg_seg = agg_seg.merge(den_total, on="bucket", how="left")
            agg_seg["pct"] = np.where(
                agg_seg["den_total"] > 0,
                (agg_seg["num_seg"] / agg_seg["den_total"]),
                np.nan,
            )

            # Línea total del bucket
            total = (
                df_c.groupby("bucket", as_index=False)
                .agg(
                    num_seg=("_mora_num_", "sum"),
                    den_total=("_monto_den_", "sum"),
                )
            )
            total[breakdown_col] = "Total"
            total["pct"] = np.where(
                total["den_total"] > 0,
                (total["num_seg"] / total["den_total"]),
                np.nan,
            )

            out = pd.concat([agg_seg, total], ignore_index=True)

            # Orden opcional por bucket y breakdown
            out["bucket"] = pd.Categorical(out["bucket"], categories=bucket_order, ordered=True)
            out = out.sort_values(["bucket", breakdown_col]).reset_index(drop=True)

            return out, bucket_order

        # =========================================================
        # MODO EVER -> contribución por conteo al % total
        # =========================================================
        ever_col = self.ever_col_from_tipo_mora(scenario.tipo_mora)
        if ever_col not in df.columns:
            raise KeyError(
                f"Columna ever '{ever_col}' no existe en la base transversal para tipo_mora={scenario.tipo_mora}."
            )

        df_e = df.copy()

        # Numerador del segmento = folios con ever_flag dentro del segmento
        df_e["_ever_flag_"] = (
            pd.to_numeric(df_e[ever_col], errors="coerce")
            .fillna(0)
            .gt(0)
            .astype(int)
        )

        # Numerador por segmento dentro del bucket
        agg_seg = (
            df_e.groupby(["bucket", breakdown_col], as_index=False)
            .agg(
                num_seg=("_ever_flag_", "sum"),
            )
        )

        # Denominador total del bucket = folios únicos totales del bucket
        den_total = (
            df_e.groupby("bucket", as_index=False)
            .agg(
                den_total=(self.folio_col, "nunique"),
            )
        )

        agg_seg = agg_seg.merge(den_total, on="bucket", how="left")
        agg_seg["pct"] = np.where(
            agg_seg["den_total"] > 0,
            (agg_seg["num_seg"] / agg_seg["den_total"]),
            np.nan,
        )

        # Línea total del bucket
        total = (
            df_e.groupby("bucket", as_index=False)
            .agg(
                num_seg=("_ever_flag_", "sum"),
                den_total=(self.folio_col, "nunique"),
            )
        )
        total[breakdown_col] = "Total"
        total["pct"] = np.where(
            total["den_total"] > 0,
            (total["num_seg"] / total["den_total"]),
            np.nan,
        )

        out = pd.concat([agg_seg, total], ignore_index=True)

        # Orden opcional por bucket y breakdown
        out["bucket"] = pd.Categorical(out["bucket"], categories=bucket_order, ordered=True)
        out = out.sort_values(["bucket", breakdown_col]).reset_index(drop=True)

        return out, bucket_order


    # ========================================
    # Core computaciones
    # ========================================
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
        - cosechas:
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

        if metric_mode == "cosechas":
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
        if metric_mode == "cosechas":
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
        exposure_by_cosecha: Optional[pd.DataFrame] = None,
        decimals: int = 1,
    ) -> pd.DataFrame:
        """
        Curva agregada por MOB.

        Nueva lógica preferida:
        - Si viene exposure_by_cosecha:
            calcula explícitamente la media ponderada de los % por cosecha
            usando como peso la exposición/originación de cada cohorte.
        - Si no viene exposure_by_cosecha:
            conserva el fallback anterior basado en sum(num)/sum(den).

        Esto permite que:
        - la fila resumen de la matriz,
        - la curva agregada,
        - y las curvas por breakdown
        empaten exactamente con la definición:
        promedio ponderado de % disponibles por MOB usando $ o # por cosecha.
        """

        y_col = "pct_impago_mob" if metric_mode == "cosechas" else "pct_ever_mob"

        # ============================================================
        # Camino preferido: media ponderada explícita por cohorte
        # ============================================================
        if exposure_by_cosecha is not None and not exposure_by_cosecha.empty:
            needed = {self.cohorte_col, self.mob_col, "pct"}
            if not needed.issubset(set(agg_long.columns)):
                raise KeyError(
                    f"agg_long debe traer columnas {sorted(needed)} para el cálculo ponderado explícito."
                )

            expo = exposure_by_cosecha.copy()
            if self.cohorte_col not in expo.columns or "exposure" not in expo.columns:
                raise KeyError(
                    "exposure_by_cosecha debe traer columnas: [cosecha, exposure]."
                )

            # Normalización de cohorte para alinear contra agg_long
            expo["_cohorte_dt_"] = pd.to_datetime(expo[self.cohorte_col], errors="coerce")
            expo = expo.loc[expo["_cohorte_dt_"].notna()].copy()

            # Si hubiera repetidos por cohorte, consolidar
            expo = (
                expo.groupby("_cohorte_dt_", as_index=False)["exposure"]
                .sum()
            )

            weights = pd.Series(
                expo["exposure"].astype(float).values,
                index=expo["_cohorte_dt_"]
            )

            base = agg_long[[self.cohorte_col, self.mob_col, "pct"]].copy()
            base["_cohorte_dt_"] = pd.to_datetime(base[self.cohorte_col], errors="coerce")
            base[self.mob_col] = pd.to_numeric(base[self.mob_col], errors="coerce")
            base["pct"] = pd.to_numeric(base["pct"], errors="coerce")
            base = base.dropna(subset=["_cohorte_dt_", self.mob_col]).copy()
            base[self.mob_col] = base[self.mob_col].astype(int)

            # Pivot cohorte x MOB con % por celda
            pct_mat = (
                base.pivot(index="_cohorte_dt_", columns=self.mob_col, values="pct")
                .sort_index()
            )

            # Alinear pesos a las filas de la matriz
            weights = weights.reindex(pct_mat.index)

            rows = []
            for mob in sorted(pct_mat.columns):
                s_pct = pd.to_numeric(pct_mat[mob], errors="coerce")
                s_w = pd.to_numeric(weights, errors="coerce")

                mask = s_pct.notna() & s_w.notna() & (s_w > 0)
                if mask.any():
                    pct_weighted = np.average(
                        s_pct.loc[mask].astype(float).values,
                        weights=s_w.loc[mask].astype(float).values,
                    )
                else:
                    pct_weighted = np.nan

                rows.append((int(mob), round(float(pct_weighted), decimals) if pd.notna(pct_weighted) else np.nan))

            out = pd.DataFrame(rows, columns=[self.mob_col, y_col]).sort_values(self.mob_col)
            return out

        # ============================================================
        # Fallback: lógica anterior sum(num)/sum(den)
        # ============================================================
        if metric_mode == "cosechas":
            needed = {"bgi_sum", "monto_sum", self.mob_col}
            if not needed.issubset(set(agg_long.columns)):
                raise KeyError(f"agg_long no trae columnas para cosechas: {sorted(needed)}")

            g = (
                agg_long.groupby(self.mob_col, as_index=False)
                .agg(bgi_sum=("bgi_sum", "sum"), monto_sum=("monto_sum", "sum"))
            )
            den = g["monto_sum"].astype(float)
            num = g["bgi_sum"].astype(float)

            g[y_col] = np.where(
                den > 0,
                (num / den) * 100.0,
                np.nan,
            ).round(decimals)

            return g.sort_values(self.mob_col)

        needed = {"ever_sum", "folio_den", self.mob_col}
        if not needed.issubset(set(agg_long.columns)):
            raise KeyError(f"agg_long no trae columnas para ever: {sorted(needed)}")

        g = (
            agg_long.groupby(self.mob_col, as_index=False)
            .agg(ever_sum=("ever_sum", "sum"), folio_den=("folio_den", "sum"))
        )
        den = g["folio_den"].astype(float)
        num = g["ever_sum"].astype(float)

        g[y_col] = np.where(
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

    def compute_originations_count_by_cosecha(self, df_filtered: pd.DataFrame) -> pd.DataFrame:
        """
        Conteo de originaciones por cosecha (formato cliente):
        originations = #folios únicos SOLO para registros con MOB == 1,
        después de aplicar filtros (y castigo si corresponde).

        Retorna columnas: [cosecha, exposure] para reusar make_matrix_display sin cambios grandes.
        """
        if self.mob_col not in df_filtered.columns:
            raise KeyError(f"No existe columna MOB '{self.mob_col}' en df_filtered.")
        if self.folio_col not in df_filtered.columns:
            raise KeyError(f"No existe columna folio '{self.folio_col}' en df_filtered.")
        if self.cohorte_col not in df_filtered.columns:
            raise KeyError(f"No existe columna cohorte '{self.cohorte_col}' en df_filtered.")

        df_mob1 = df_filtered.loc[df_filtered[self.mob_col] == 1].copy()

        g = (
            df_mob1.groupby(self.cohorte_col, as_index=False)
            .agg(exposure=(self.folio_col, "nunique"))
        )
        g["_cosecha_dt"] = pd.to_datetime(g[self.cohorte_col], errors="coerce")
        g = g.sort_values("_cosecha_dt").drop(columns=["_cosecha_dt"])

        return g

    def compute_curves_by_mob_breakdown(
        self,
        scenario: Scenario,
        *,
        breakdown_col: str = "Perfil",
        max_levels: int = None,
        min_folios: int = 200,
        mob_max: Optional[int] = None,
    ) -> list[dict]:
        """
        Compute-only: regresa lista de curves para breakdown (sin plot).
        Cada item: {"label": str, "curve": pd.DataFrame, "last_val": float|nan}
        """

        if breakdown_col not in self.df.columns:
            raise KeyError(f"breakdown_col '{breakdown_col}' no existe en df.")

        # Base filtrada por escenario
        df_base = self.apply_filters(self.df, scenario.filters).copy()
        if df_base.empty:
            raise ValueError(
                f"Escenario '{scenario.name}' dejó 0 filas tras filtros: {scenario.filters}."
            )

        # Castigo global (switch)
        df_base = self.apply_castigo_filter(df_base).copy()
        if df_base.empty:
            raise ValueError(
                f"Escenario '{scenario.name}' quedó vacío tras castigo_enabled={self.castigo_enabled} "
                f"({self.castigo_col} excluye {self.castigo_exclude_values})."
            )

        # Niveles candidatos
        df_base = self._drop_missing_breakdown_rows(df_base, breakdown_col)
        if df_base.empty:
            raise ValueError(
                f"No hay registros con dato válido en '{breakdown_col}' bajo filtros={scenario.filters}."
            )

        levels = self._get_valid_breakdown_levels(df_base, breakdown_col)

        # Soporte por nivel (#folios)
        level_stats = []
        for lv in levels:
            df_lv = df_base.loc[df_base[breakdown_col].astype(str).str.strip().eq(lv)]
            n_folios = df_lv[self.folio_col].nunique()
            if n_folios >= min_folios:
                level_stats.append((lv, n_folios))

        if not level_stats:
            raise ValueError(
                f"No hay niveles en '{breakdown_col}' con >= {min_folios} folios bajo filtros={scenario.filters}."
            )

        # Orden negocio (YAML) para elegir max_levels
        order_list = getattr(self, "legend_order_map", {}).get(breakdown_col, None)
        if isinstance(order_list, list) and order_list:
            rank = {str(v).strip(): i for i, v in enumerate(order_list)}
            level_stats.sort(
                key=lambda t: (
                    0 if t[0] in rank else 1,
                    rank.get(t[0], 10**9),
                    -t[1],
                    t[0],
                )
            )
        else:
            level_stats.sort(key=lambda t: (-t[1], t[0]))

        if max_levels is not None:
            level_stats = level_stats[:max_levels]

        # Columna y según modo
        y_col = "pct_impago_mob" if scenario.metric_mode == "cosechas" else "pct_ever_mob"

        # Computa curves
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

            if scenario.metric_mode == "cosechas":
                exposure_sub = self.compute_exposure_by_cosecha(df_lv)
            else:
                exposure_sub = self.compute_originations_count_by_cosecha(df_lv)

            curve_sub = self.compute_curve_by_mob(
                agg_sub,
                metric_mode=scenario.metric_mode,
                exposure_by_cosecha=exposure_sub,
            )

            s = curve_sub[y_col].dropna()
            last_val = s.iloc[-1] if not s.empty else np.nan

            curves.append(
                {
                    "label": f"{lv}",
                    "curve": curve_sub,
                    "last_val": float(last_val) if pd.notna(last_val) else np.nan,
                }
            )

        # Orden final de leyenda: YAML si existe; si no, last_val
        if isinstance(order_list, list) and order_list:
            rank = {str(v).strip(): i for i, v in enumerate(order_list)}

            def sort_key(d):
                lab = str(d["label"]).strip()
                return (0, rank[lab]) if lab in rank else (1, lab)

            curves = sorted(curves, key=sort_key)
        else:
            curves = sorted(curves, key=lambda d: (np.isnan(d["last_val"]), d["last_val"]))

        # Recorte por mob_max (a cada curva)
        if mob_max is not None:
            for it in curves:
                c = it["curve"].copy()
                mob_num = pd.to_numeric(c[self.mob_col], errors="coerce")
                mask = mob_num.notna() & (mob_num <= mob_max)
                c = c.loc[mask].copy()
                c[self.mob_col] = mob_num.loc[mask].astype(int)
                it["curve"] = c

        return curves


    # ========================================
    # Plotters
    # ========================================
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
            [d.strftime("%y-%m") if pd.notna(d) else "NaT" for d in m.index]
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
        mob_max: Optional[int] = None,
        show: bool = True,
        debug: Optional[bool] = None,
        show_point_labels: bool = False,
    ) -> plt.Figure:
        """
        (2) Curvas desagregadas por MOB, respetando filtros del escenario.
        """

        # Suavizador de curva (solo visual)
        def _smooth(y, span=4.5):
            s = pd.Series(y).astype(float)
            return s.ewm(span=span, adjust=False).mean().values

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
        df_base = self._drop_missing_breakdown_rows(df_base, breakdown_col)
        if df_base.empty:
            raise ValueError(
                f"No hay registros con dato válido en '{breakdown_col}' bajo filtros={scenario.filters}."
            )

        levels = self._get_valid_breakdown_levels(df_base, breakdown_col)

        # Soporte por nivel (#folios)
        level_stats = []
        for lv in levels:
            df_lv = df_base.loc[df_base[breakdown_col].astype(str).str.strip().eq(lv)]
            n_folios = df_lv[self.folio_col].nunique()
            if n_folios >= min_folios:
                level_stats.append((lv, n_folios))

        if not level_stats:
            raise ValueError(
                f"No hay niveles en '{breakdown_col}' con >= {min_folios} folios bajo filtros={scenario.filters}."
            )

        # --- Orden de negocio (si existe) para elegir max_levels
        order_list = getattr(self, "legend_order_map", {}).get(breakdown_col, None)
        if isinstance(order_list, list) and order_list:
            rank = {str(v).strip(): i for i, v in enumerate(order_list)}
            level_stats.sort(
                key=lambda t: (
                    0 if t[0] in rank else 1,
                    rank.get(t[0], 10**9),
                    -t[1],
                    t[0],
                )
            )
        else:
            level_stats.sort(key=lambda t: (-t[1], t[0]))

        level_stats = level_stats[:max_levels]

        # Figura con ax
        fig, ax = plt.subplots(figsize=(12, 7))

        # Columna y según modo
        y_col = "pct_impago_mob" if scenario.metric_mode == "cosechas" else "pct_ever_mob"

        # 1) Computar curvas
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

            if scenario.metric_mode == "cosechas":
                exposure_sub = self.compute_exposure_by_cosecha(df_lv)
            else:
                exposure_sub = self.compute_originations_count_by_cosecha(df_lv)

            curve_sub = self.compute_curve_by_mob(
                agg_sub,
                metric_mode=scenario.metric_mode,
                exposure_by_cosecha=exposure_sub,
            )

            s = curve_sub[y_col].dropna()
            last_val = s.iloc[-1] if not s.empty else np.nan

            curves.append(
                {
                    "label": f"{lv}",
                    "curve": curve_sub,
                    "last_val": float(last_val) if pd.notna(last_val) else np.nan,
                }
            )

        # 2) Orden de leyenda por YAML si existe (si no, fallback por last_val)
        if isinstance(order_list, list) and order_list:
            rank = {str(v).strip(): i for i, v in enumerate(order_list)}

            def sort_key(d):
                lab = str(d["label"]).strip()
                return (0, rank[lab]) if lab in rank else (1, lab)

            curves = sorted(curves, key=sort_key)
        else:
            curves = sorted(curves, key=lambda d: (np.isnan(d["last_val"]), d["last_val"]))

        # 3) Recorte por mob_max (a cada curva)
        if mob_max is not None:
            for it in curves:
                c = it["curve"].copy()
                mob_num = pd.to_numeric(c[self.mob_col], errors="coerce")
                mask = mob_num.notna() & (mob_num <= mob_max)

                c = c.loc[mask].copy()
                c[self.mob_col] = mob_num.loc[mask].astype(int)

                it["curve"] = c

        # 4) Plot (sin markers, suavizado)
        for it in curves:
            c = it["curve"].copy()
            mob_num = pd.to_numeric(c[self.mob_col], errors="coerce")
            y_num = pd.to_numeric(c[y_col], errors="coerce")

            mask = mob_num.notna() & y_num.notna()
            if mob_max is not None:
                mask = mask & (mob_num <= mob_max)

            x = mob_num.loc[mask].astype(int).values
            y = y_num.loc[mask].astype(float).values
            y_smooth = _smooth(y, span=5)

            line, = ax.plot(
                x,
                y_smooth,
                linewidth=1.5,
                label=it["label"]
            )
            line_color = line.get_color()

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
                        color=line_color,
                    )
                    txt.set_path_effects([
                        path_effects.Stroke(linewidth=3, foreground="white"),
                        path_effects.Normal()
                    ])

        # Eje X: todos los MOBs + límites estrictos
        if mob_max is not None:
            ax.set_xlim(0, mob_max + 1)
            ax.set_xticks(range(1, mob_max + 1))
            plt.setp(ax.get_xticklabels(), rotation=0)

        ax.set_xlabel("MOB")

        # Label Y tipo %2+, %3+, %4+, %5+, %Cast
        if scenario.tipo_mora.startswith("BGI"):
            nivel = scenario.tipo_mora.replace("BGI", "")
            y_label = f"%{nivel}"
        elif scenario.tipo_mora == "CAST":
            y_label = "%Cast"
        else:
            y_label = "%"

        ax.set_ylabel(y_label)
        ax.set_title(f"Curvas por MOB (desagregado por {breakdown_col}) – {scenario.name}")

        # Solo grid horizontal
        ax.grid(False)
        ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=1.0)

        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        if show:
            plt.show()

        return fig


    # ========================================
    # IO helpers
    # ========================================
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
        - primera columna (opcional): $ (monto) o # (conteo) por cosecha (MOB==1)
        - celdas: % con cell_decimals
        - fila resumen por MOB: % con summary_decimals usando compute_curve_by_mob()
        """
        m_num = matrix_dt.copy()

        # --- index YYYY-MM
        idx_str = [d.strftime("%y-%m") if pd.notna(d) else blank for d in m_num.index]
        m_num.index = idx_str

        def _fmt_cell(x):
            if pd.isna(x):
                return blank
            return f"{float(x):.{cell_decimals}f}"

        def _fmt_summary(x):
            if pd.isna(x):
                return blank
            return f"{float(x):.{summary_decimals}f}"

        # Matriz principal (celdas)
        m_disp = m_num.map(_fmt_cell)

        # --- insertar columna extra ($ o #) si viene exposure_by_cosecha
        if exposure_by_cosecha is not None:
            if self.cohorte_col not in exposure_by_cosecha.columns or "exposure" not in exposure_by_cosecha.columns:
                raise KeyError("exposure_by_cosecha debe traer columnas: [cosecha, exposure].")

            expo = exposure_by_cosecha.copy()
            expo["_cosecha_dt"] = pd.to_datetime(expo[self.cohorte_col], errors="coerce")
            expo = expo.sort_values("_cosecha_dt").drop(columns=["_cosecha_dt"])

            expo_idx = [
                pd.to_datetime(x, errors="coerce").strftime("%y-%m")
                if pd.notna(pd.to_datetime(x, errors="coerce"))
                else blank
                for x in expo[self.cohorte_col]
            ]

            expo_series = pd.Series(expo["exposure"].values, index=expo_idx)

            def _fmt_money(v):
                if pd.isna(v):
                    return blank
                try:
                    return f"{float(v):,.0f}"
                except Exception:
                    return blank

            def _fmt_count(v):
                if pd.isna(v):
                    return blank
                try:
                    return f"{int(round(float(v), 0))}"
                except Exception:
                    return blank

            fmt_func = _fmt_money if exposure_col_label == "$" else _fmt_count
            m_disp.insert(0, exposure_col_label, expo_series.reindex(m_disp.index).map(fmt_func))

        # fila en blanco
        if add_blank_row:
            empty_row = pd.DataFrame([[blank] * m_disp.shape[1]], columns=m_disp.columns, index=[blank])
            m_disp = pd.concat([m_disp, empty_row], axis=0)

        # fila resumen por MOB usando compute_curve_by_mob
        if add_mob_summary_row:
            if agg_long is None or metric_mode is None:
                raise ValueError("add_mob_summary_row=True requiere agg_long y metric_mode.")

            curve_df = self.compute_curve_by_mob(
                agg_long,
                metric_mode=metric_mode,
                exposure_by_cosecha=exposure_by_cosecha,
                decimals=summary_decimals,
            )
            y_col = "pct_impago_mob" if metric_mode == "cosechas" else "pct_ever_mob"
            summary_series = curve_df.set_index(self.mob_col)[y_col]

            # si hay columna extra ($ o #), dejamos esa celda en blanco en la fila resumen
            if exposure_col_label in m_disp.columns:
                summary_series = summary_series.reindex([c for c in m_disp.columns if c != exposure_col_label])
                summary_row = pd.DataFrame(
                    [[blank] + list(summary_series.reindex(m_disp.columns[1:]).values)],
                    columns=m_disp.columns,
                    index=[summary_label],
                )
            else:
                summary_row = pd.DataFrame(
                    [summary_series.reindex(m_disp.columns).values],
                    columns=m_disp.columns,
                    index=[summary_label],
                )

            # formatear solo la parte de MOBs (no tocar la columna extra)
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
        - "BGI4+" -> "Total 4+"
        - "BGI3+" -> "Total 3+"
        - "BGI2+" -> "Total 2+"
        - "BGI5+" -> "Total 5+"
        - "CAST" -> "Total CAST"
        """
        if tipo_mora == "CAST":
            return "Total CAST"
        if isinstance(tipo_mora, str) and tipo_mora.startswith("BGI") and tipo_mora.endswith("+"):
            nivel = tipo_mora.replace("BGI", "") # "BGI4+" -> "4+"
            return f"Total {nivel}"
        return f"Total {str(tipo_mora)}"

    def load_legend_order_map(self, path: Path) -> dict[str, list[str]]:
        """
        Carga el YAML con orden de leyenda.
        Devuelve dict[str, list[str]] con normalización str+strip.
        """
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        out: dict[str, list[str]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[str(k).strip()] = [str(x).strip() for x in v]
            # si v no es lista (ej. texto), lo ignoramos
        return out


    # ========================================
    # Para correr escenarios
    # ========================================
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

        # Orden interno por cosecha (sin tocar contrato): ayuda a consistencia en logs
        df_f["_cosecha_dt"] = pd.to_datetime(df_f[self.cohorte_col], errors="coerce")
        df_f = df_f.sort_values(["_cosecha_dt", self.mob_col]).drop(columns=["_cosecha_dt"])

        agg_long, matrix_dt = self.compute_matrix(
            df_f,
            tipo_mora=scenario.tipo_mora,
            metric_mode=scenario.metric_mode,
        )

        # --- Base de pesos por cohorte:
        #     $ para cosechas / # para ever
        if scenario.metric_mode == "cosechas":
            extra_df = self.compute_exposure_by_cosecha(df_f)
            extra_label = "$"
        else:
            extra_df = self.compute_originations_count_by_cosecha(df_f)
            extra_label = "#"

        curve = self.compute_curve_by_mob(
            agg_long,
            metric_mode=scenario.metric_mode,
            exposure_by_cosecha=extra_df,
        )

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
        y_col = "pct_impago_mob" if scenario.metric_mode == "cosechas" else "pct_ever_mob"
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
        x_dt = pd.to_datetime(extra_df[self.cohorte_col], errors="coerce")
        plt.plot(x_dt, extra_df["exposure"], marker="o", color=ATRIA_PURPLE)
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
            )

        plt.tight_layout()
        figs["heatmap"] = fig3
        if show:
            plt.show()

        summary_label = self.summary_label_from_tipo_mora(scenario.tipo_mora)

        matrix_display = self.make_matrix_display(
            matrix_dt,
            cell_decimals=1,
            summary_decimals=1,
            blank="",
            add_blank_row=True,
            add_mob_summary_row=True,
            summary_label=summary_label,  
            agg_long=agg_long,
            metric_mode=scenario.metric_mode,
            exposure_by_cosecha=extra_df,
            exposure_col_label=extra_label,
        )

        # (1) Matriz “como tabla”
        result: Dict[str, object] = {
            "df_filtered": df_f,
            "agg_long": agg_long,
            "matrix_dt": matrix_dt,
            "matrix_display": matrix_display,
            "curve_mob": curve,
            "exposure_by_cosecha": extra_df,

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
                exposure_by_cosecha=extra_df,
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
