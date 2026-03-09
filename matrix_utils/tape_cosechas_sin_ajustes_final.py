import pandas as pd
from datetime import datetime
from io import BytesIO
from dateutil.relativedelta import relativedelta
from etl.utils.logs import get_logger
from etl.utils.queries import (
    #get_last_loan_status,
    #get_movements_by_cutoff_date,
    get_movements_parent_id,
    get_payments_parent_id,
    get_pv_from_r2_dir_dpd,
    get_loan_investors,
    get_monthly_payment_for_fondeador
    #get_comprada_date,
    #get_formalizadas
)

logger = get_logger()

final_cols = [
    'folio',
    'MOB',
    'pv',
    'Fecha de Inicio',
    'Mes-Año',
    'Monto Fondeado',
    'Saldo Capital',
    'Tipo de Castigo',
    'Estatus Legal',
    'BGI4+',
    'Finan',
    'Mob Castigo',
    'Trimestre',
    'BGI4+_MAX',
    'BGI4+_CONTEO',
    'BGI4+_CONTEO_EVER',
    'CONTEO_FINAN',
    'BGI2+',
    'BGI3+',
    'BGI5+',
    'BGI2+_CONTEO',
    'BGI3+_CONTEO',
    'BGI5+_CONTEO',
    'BGI2+_CONTEO_EVER',
    'BGI3+_CONTEO_EVER',
    'BGI5+_CONTEO_EVER',
    'CAST_VAL',
    'CAST_CONTEO',
    'CAST_VAL_ORI',
]


def get_total_pay_by_folio(mysqlCon):

    logger.info("Obteniendo pagado total")
    logger.info(f"get_total_pay_by_folio init")

    query = """
    select l.folio, p.payment_date,
    p.amount pago_total,
    pp.accounting_movement_id parent_id
    from payments p 
    inner join loans l on l.payment_control_id = p.payment_control_id
    left join payment_priorities pp on pp.payment_id = p.id
    and p.amount != 0
    """

    with mysqlCon.connect() as connection:
        total_pay_table = pd.read_sql(query, con=connection)

    logger.info(f"get_total_pay_by_folio finished")
    
    return total_pay_table


def get_capital_by_folio(mysqlCon):

    logger.info("Obteniendo esperado y pagado de capital")
    logger.info(f"get_capital_by_folio init")
    
    query = """
    select l.folio, l.created_at, am.parent_id, am.id, 
    max(if(am.accounting_movement_type_id = 1, ast.cutoff_date,'')) cutoff_date,
    am.movement_date, min(ast.term) term, 
    sum(if(am.accounting_movement_type_id = 1, am.amount, 0)) cargo,
    sum(if(am.accounting_movement_type_id = 2, am.amount, 0)) abonos
    from accounting_movements am 
    inner join account_statements `ast` on am.account_statement_id = `ast`.id 
    inner join loans l on l.accounting_control_id = am.accounting_control_id 
    where am.accounting_movement_type_id in (1, 2)
    and am.accounting_account_id = 1001001
    group by l.folio, am.accounting_account_id, am.movement_date, am.parent_id
    """
    
    with mysqlCon.connect() as connection:
        capital_table = pd.read_sql(query, con=connection)

    logger.info(f"get_capital_by_folio finished")
    
    return capital_table


#Esta query devuelve los abonos a capital
#en accounting_account_id2 indica si fue castigo, pago real, gasto, etc.
def get_capital_paid(mysqlCon):
    
    logger.info("Obteniendo capital pagado")
    logger.info(f"get_capital_paid init")
    
    query = """
    select l.folio, am.id, am.movement_date, am.amount, am.parent_id,
    am2.parent_id parent_id2, am2.accounting_account_id accounting_account_id2,
    p.payment_date
    #pp.payment_id id_pago_original, p.amount
    from accounting_movements am 
    inner join loans l on l.accounting_control_id = am.accounting_control_id 
    inner join accounting_movements am2 on am2.id = am.parent_id 
    left join payment_priorities pp on pp.accounting_movement_id = am2.parent_id
    left join payments p on p.id = pp.payment_id
    where am.accounting_movement_type_id in (2)
    and am.accounting_account_id = 1001001
    and am.amount != 0
    """
    
    #query = """
    #select l.folio, am.id, am.movement_date, am3.movement_date fecha_pago, am.amount, am.parent_id,
    #am2.parent_id parent_id2, am2.accounting_account_id accounting_account_id2
    #from accounting_movements am 
    #inner join loans l on l.accounting_control_id = am.accounting_control_id 
    #inner join accounting_movements am2 on am2.id = am.parent_id 
    #inner join accounting_movements am3 on am3.id = am.parent_id 
    #where am.accounting_movement_type_id in (2)
    #and am.accounting_account_id = 1001001
    #and am.amount != 0
    #"""
    
    with mysqlCon.connect() as connection:
        capital_paid_table = pd.read_sql(query, con=connection)

    logger.info(f"get_capital_paid finished")
    
    return capital_paid_table


def get_month_year(df,col,t=None,col_name='Mes-Año'):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df['year'] = df[col].dt.year
    df['month'] = df[col].dt.month
    
    if t == 1:
        df[col_name] = df.apply(lambda x: str(x['month']) + '-' + str(x['year']), axis = 1)
    
    return df


def get_min_BG(df,col):

    logger.info(f"Calculando EVER para {col}")
    
    pv_ever = df.copy()
    pv_ever = pv_ever[pv_ever[col]==1]
    pv_ever = pv_ever.groupby(['folio',col]).agg({'MOB':'min'}).reset_index()
    pv_ever.rename(columns={"MOB": "min_MOB_"+str(col)}, inplace=True)

    df = pd.merge(df, pv_ever[['folio',"min_MOB_"+str(col)]], on=['folio'], how="left")
    
    return df
    

def create_tape_cosechas(mysqlCon, etlCon, date=None, col_date='movement_date'):

    min_date = datetime.strptime("2022-07-01", "%Y-%m-%d")
    
    if date == None:
        last_date = pd.to_datetime('today')
        last_date = last_date.strftime('%Y-%m-%d %H:%M:%S')
    else:
        last_date = pd.to_datetime(date)

    #Consultas a la base
    logger.info("Obteniendo datos...")
    expected = get_capital_by_folio(mysqlCon)   #Leyendo datos de capital esperado
    capital_paid = get_capital_paid(mysqlCon)   #Leyendo datos de capital recibido
    monthly_pay = get_monthly_payment_for_fondeador(mysqlCon)   #Leyendo datos de mensualidades
    pv_data = get_pv_from_r2_dir_dpd(etlCon)   #Leyendo datos de pv
    folios_investor = get_loan_investors(mysqlCon)   #Leyendo datos de inversionista
    total_payments = get_total_pay_by_folio(mysqlCon)   #Pagos completos (sin prelar) para castigos

    #Agregar datos manuales de castigos (temporal)
    logger.info("Agregando castigos...")
    castigos_data = pd.read_excel('castigos_cosechas_20251202.xlsx', index_col=None)
    castigos_data['Folio'] = castigos_data['Folio'].astype(int)
    castigos_data['Folio'] = castigos_data['Folio'].astype(str)

    #Agregar datos manuales de castigos (temporal)
    logger.info("Agregando datos pv...")
    pv_fix = pd.read_excel('pv_gaby.xlsx', index_col=None)
    pv_fix['folio'] = pv_fix['folio'].astype(int)
    pv_fix['folio'] = pv_fix['folio'].astype(str)
    
    #Filtrando folios de creditas
    folios_cred = folios_investor[folios_investor['investor'] == "CREDITAS"]
    list_folios = folios_cred['folio'].tolist()
    
    #Generar meses desde el primer mes de originación
    logger.info("Generando meses faltantes...")
    all_months = pd.date_range(start=min_date, end=last_date, freq='ME').to_frame()
    all_months.reset_index(inplace=True)
    all_months.rename(columns={'index': 'dates'}, inplace=True)
    all_months = all_months['dates'].to_frame()
    all_months = get_month_year(all_months,'dates')
    
    #obtener el primer corte de cada folio
    extreme_cuts = expected.copy()
    extreme_cuts = extreme_cuts[extreme_cuts['cutoff_date'] != ""]
    
    extreme_cuts = (extreme_cuts.groupby('folio').agg(
        cutoff_date_min=('cutoff_date', 'min'),
        cutoff_date_max=('cutoff_date', 'max')
    ).reset_index())

    extreme_cuts = get_month_year(extreme_cuts,'cutoff_date_min',1,'Mes-Año_min')
    extreme_cuts = get_month_year(extreme_cuts,'cutoff_date_max',1,'Mes-Año_max') 

    capital_total = expected.groupby(['folio']).agg({"cargo":'sum'}).reset_index()
    capital_total.rename(columns={"cargo": "Monto Fondeado"}, inplace=True)
    
    #Obteniendo la fecha de formalización
    start_date_table = expected[['folio','created_at']]
    start_date_table = start_date_table.drop_duplicates()
    start_date_table.rename(columns={'created_at': 'Fecha Inicio'}, inplace=True)
    start_date_table.sort_values(by=["folio"], inplace=True)
    
    start_date_table["first_month"] = start_date_table.apply(
        lambda x: pd.to_datetime(x['Fecha Inicio'].strftime("%Y-%m-01")) + relativedelta(months=1),
        axis = 1
    )

    start_date_table["fecha_trimestre"] = pd.to_datetime(start_date_table['Fecha Inicio'])
    start_date_table["Trimestre"] = start_date_table['fecha_trimestre'].dt.to_period('Q').dt.strftime('%Y-T%q')
    

    """ INTENTO 2: CÁLCULO FECHA CASTIGO """
    castigos_data_v2 = castigos_data.copy()
    castigos_data_v2.drop(['Fecha_castigo','Fecha_castigo_2'], axis=1)
    castigos_data_v2 = pd.merge(
        castigos_data_v2,
        start_date_table[['folio','Fecha Inicio']],
        left_on='Folio',
        right_on='folio',
        how='left'
    )
    castigos_data_v2["Fecha_castigo"] = castigos_data_v2.apply(
        lambda x: pd.NaT if pd.isna(x["Mob Castigo"]) 
        else (x["Fecha Inicio"] + relativedelta(months=(int(x["Mob Castigo"]) + 1))).replace(day=1),
        axis = 1
    )
    
    castigos_data_v2["Fecha_castigo"] = castigos_data_v2["Fecha_castigo"].dt.normalize()
    """ FIN INTENTO 2: CÁLCULO FECHA CASTIGO """
    
    """ CÁLCULO DE PAGOS POSTERIORES A CASTIGO """
    total_payments = pd.merge(
        total_payments,
        castigos_data_v2[['folio','Fecha_castigo']],
        on='folio',
        how='left'
    )

    total_payments = total_payments[total_payments['Fecha_castigo'].notna()]
    total_payments['Fecha_castigo'] = pd.to_datetime(total_payments['Fecha_castigo'])
    total_payments['payment_date'] = pd.to_datetime(total_payments['payment_date'])
    total_payments['check'] = total_payments.apply(
        lambda x: 1 if x["payment_date"] >= x["Fecha_castigo"]
        else 0,
        axis = 1
    )

    total_payments.to_csv('total_payments_v0.csv')
    total_payments = total_payments[total_payments['check'] == 1]

    total_payments = get_month_year(total_payments,'payment_date')

    total_payments = (total_payments.groupby(["folio","year","month"]).agg({"pago_total": "sum"}).reset_index())
    total_payments.sort_values(by=["year","month","folio"], inplace=True)
    
    total_payments.to_csv('total_payments_v1.csv')
    """ FIN CÁLCULO DE PAGOS POSTERIORES A CASTIGO """
    
    logger.info("Calculando Saldo Capital...")
    #Filtrar los abonos que son de castigo
    capital_paid = capital_paid[~capital_paid['accounting_account_id2'].isin(
        [1011002,
         1011003,
         1011009,
         #1011016, #bienes adjudicados (se agrega para cuadre de reporte)
         1011005  #condonación (se agrega para cuadre de reporte)
        ])] 

    capital_paid_fixed = pd.merge(capital_paid, start_date_table, on='folio', how='left')
    capital_paid_fixed['first_month'] = pd.to_datetime(capital_paid_fixed['first_month'])

    #Se agrega para poder cambiar entre fecha de aplicación o fecha en que se recibio el pago
    if col_date == "payment_date":
        capital_paid_fixed[col_date] = capital_paid_fixed.apply(
            lambda x: x['movement_date'] if ((pd.isna(x["parent_id2"])) or (x['accounting_account_id2'] in (1009006,1009007))) else x[col_date],
            axis = 1
        )
    
    capital_paid_fixed[col_date] = pd.to_datetime(capital_paid_fixed[col_date])
    capital_paid_fixed['movement_date_normalize'] = capital_paid_fixed.apply(
        lambda x: x['first_month'] if x[col_date] < x['first_month'] else x[col_date],
        axis = 1
    )
    
    capital_paid_fixed = get_month_year(capital_paid_fixed,'movement_date_normalize')
    
    capital_paid_1 = (capital_paid_fixed.groupby(["folio","year","month"]).agg({"amount": "sum"}).reset_index())
    capital_paid_1.sort_values(by=["year","month","folio"], inplace=True)
    

    folios_mes = pd.merge(all_months, capital_total, on=None, how="cross")
    folios_mes.sort_values(by=["folio","year","month"], inplace=True)

    final_table = pd.merge(folios_mes, capital_paid_1, on=["folio","year","month"], how="left")
    final_table = pd.merge(final_table, start_date_table[['folio','Fecha Inicio','first_month','Trimestre']], on=["folio"], how="left")
    
    final_table['check'] = final_table.apply(
        lambda x: 0 if x['dates'] < x['first_month'] else 1,
        axis = 1
    )
    final_table = final_table[final_table['check'] == 1]
    final_table['amount'] = final_table['amount'].fillna(0)

    """ AGREGAR DATOS DE PAGOS POSTERIORES A CASTIGOS """
    final_table = pd.merge(
        final_table,
        total_payments,
        on=['folio','year','month'],
        how='left'
    )

    final_table["amount_v2"] = final_table.apply(
        lambda x: x['amount'] if pd.isna(x['pago_total'])
        else x['pago_total'],
        axis = 1
    )
    """ FIN AGREGAR DATOS DE PAGOS POSTERIORES A CASTIGOS """

    #Versión con pagos completos por castigo
    final_table["acum_abonos"] = final_table.groupby(["folio"])["amount_v2"].agg(
            "cumsum"
        )

    final_table["Saldo Capital"] = final_table["Monto Fondeado"] - final_table["acum_abonos"]
    final_table["Counter"] = 1
    final_table["MOB"] = final_table.groupby(["folio"])["Counter"].agg(
            "cumsum"
        )
    final_table["Mes-Año"] = final_table.apply(lambda x: str(x["month"]) + '-' + str(x["year"]), axis = 1)
    final_table["Fecha de Inicio"] = final_table.apply(
        lambda x: pd.to_datetime(x['Fecha Inicio'].strftime("%Y-%m")),
        axis = 1
    )

    #Quitando folios de CREDITAS
    final_table = final_table[~final_table['folio'].isin(list_folios)]

    #agregar PV de la tabla r2_dir_dpd
    pv_data = get_month_year(pv_data,'cutoff_date',1)
    
    final_table = pd.merge(final_table, pv_data[['folio','pv','Mes-Año']], on=['folio','Mes-Año'], how="left")

    #agregar datos de castigos
    final_table = pd.merge(final_table, castigos_data, left_on=['folio'], right_on=['Folio'], how="left")
    
    cols = ['Tipo de Castigo','Estatus Legal','Mob Castigo']

    final_table = pd.merge(final_table, extreme_cuts[['folio','Mes-Año_min','Mes-Año_max']], on=['folio'], how="left")

    final_table["Mes-Año_comp"] = pd.to_datetime(final_table["Mes-Año"], format="%m-%Y", errors='coerce')
    final_table["Mes-Año_min_comp"] = pd.to_datetime(final_table["Mes-Año_min"], format="%m-%Y", errors='coerce')
    final_table["Mes-Año_max_comp"] = pd.to_datetime(final_table["Mes-Año_max"], format="%m-%Y", errors='coerce')

    final_table["pv"] = final_table.apply(lambda x: 7 if ((x["MOB"] >= x["Mob Castigo"]) and (x["Mob Castigo"] != "")) else x["pv"], axis = 1) #PV para castigos
    final_table["pv"] = final_table.apply(lambda x: 0 if ((x["Saldo Capital"] < 10) and pd.isna(x["pv"])) else x["pv"], axis = 1) #PV para meses con Saldo Capital cercano a 0

    final_table["pv"] = final_table.apply(lambda x: 0 if 
                                          ((x["Mes-Año_comp"] < x["Mes-Año_min_comp"])
                                           and pd.isna(x["pv"])) else x["pv"],
                                          axis = 1) #PV para meses de gracia
    
    #Proceso para llenar los pv faltantes
    last_pv = final_table.copy()
    last_pv = last_pv[last_pv['Mes-Año_comp'] == last_pv['Mes-Año_max_comp']]
    last_pv = last_pv[['folio','Saldo Capital','pv']]
    last_pv.rename(columns={"Saldo Capital": "last_Saldo Capital","pv": "last_pv"}, inplace=True)

    final_table = pd.merge(final_table, last_pv, on=['folio'], how="left")

    final_table["pv"] = final_table.apply(lambda x: x["last_pv"] if 
                                          ((x["Mes-Año_comp"] > x["Mes-Año_max_comp"]) 
                                           and (x["Saldo Capital"] == x["last_Saldo Capital"])
                                           and pd.isna(x["pv"])) else x["pv"],
                                          axis = 1) #PV para meses posteriores al cierre del crédito
    
    final_table["pv"] = final_table.apply(lambda x: 7 if x["pv"] > 7 else x["pv"], axis = 1) #PV para meses con PV mayor a 7

    final_table["pv"] = final_table.apply(lambda x: 4 if 
                                          ((x["folio"] == '111120230600232')
                                           and pd.isna(x["pv"])) else x["pv"],
                                          axis = 1) #PV para meses posteriores al cierre del crédito

    final_table["pv"] = final_table["pv"].fillna(0)
    
    final_table["Saldo Capital Final"] = final_table.apply(lambda x: 0 if x["Saldo Capital"] <= 0 else x["Saldo Capital"], axis = 1)

    final_table["BGI2+"] = final_table.apply(lambda x: x["Saldo Capital Final"] if x["pv"] >= 2 else "", axis = 1)
    final_table["BGI3+"] = final_table.apply(lambda x: x["Saldo Capital Final"] if x["pv"] >= 3 else "", axis = 1)
    final_table["BGI4+"] = final_table.apply(lambda x: x["Saldo Capital Final"] if x["pv"] >= 4 else "", axis = 1)
    final_table["BGI5+"] = final_table.apply(lambda x: x["Saldo Capital Final"] if x["pv"] >= 5 else "", axis = 1)
    final_table["CAST_VAL"] = final_table.apply(lambda x: x["Saldo Capital Final"] if x["pv"] >= 7 else "", axis = 1)

    final_table["BGI2+_CONTEO"] = final_table.apply(lambda x: 1 if x["pv"] >= 2 else 0, axis = 1)
    final_table["BGI3+_CONTEO"] = final_table.apply(lambda x: 1 if x["pv"] >= 3 else 0, axis = 1)
    final_table["BGI4+_CONTEO"] = final_table.apply(lambda x: 1 if x["pv"] >= 4 else 0, axis = 1)
    final_table["BGI5+_CONTEO"] = final_table.apply(lambda x: 1 if x["pv"] >= 5 else 0, axis = 1)
    final_table["CAST_CONTEO"] = final_table.apply(lambda x: 1 if x["pv"] >= 7 else 0, axis = 1)

    final_table = get_min_BG(final_table,'BGI2+_CONTEO')
    final_table = get_min_BG(final_table,'BGI3+_CONTEO')
    final_table = get_min_BG(final_table,'BGI4+_CONTEO')
    final_table = get_min_BG(final_table,'BGI5+_CONTEO')
    
    final_table["BGI2+_CONTEO_EVER"] = final_table.apply(lambda x: 1 if x["MOB"] >= x["min_MOB_BGI2+_CONTEO"] else 0, axis = 1)
    final_table["BGI3+_CONTEO_EVER"] = final_table.apply(lambda x: 1 if x["MOB"] >= x["min_MOB_BGI3+_CONTEO"] else 0, axis = 1)
    final_table["BGI4+_CONTEO_EVER"] = final_table.apply(lambda x: 1 if x["MOB"] >= x["min_MOB_BGI4+_CONTEO"] else 0, axis = 1)
    final_table["BGI5+_CONTEO_EVER"] = final_table.apply(lambda x: 1 if x["MOB"] >= x["min_MOB_BGI5+_CONTEO"] else 0, axis = 1)

    max_bgi4 = final_table.copy()
    max_bgi4 = max_bgi4[max_bgi4['BGI4+_CONTEO_EVER']==1]
    max_bgi4 = max_bgi4.groupby(['folio']).agg({'Saldo Capital Final':'max'}).reset_index()
    max_bgi4.rename(columns={"Saldo Capital Final": "Saldo_BGI4+_MAX"}, inplace=True)
    
    final_table = pd.merge(final_table, max_bgi4[['folio','Saldo_BGI4+_MAX']], on=['folio'], how="left")
    
    final_table["BGI4+_MAX"] = final_table.apply(lambda x: 0 if x["BGI4+_CONTEO_EVER"] == 0 else x["Saldo_BGI4+_MAX"], axis = 1)

    #NUEVO 2025-11-24: Agregar monto original de castigo
    ori_cast = final_table.copy()
    ori_cast = ori_cast[ori_cast['MOB']==ori_cast['Mob Castigo']]
    ori_cast.rename(columns={"Saldo Capital": "CAST_VAL_ORI"}, inplace=True)

    final_table = pd.merge(final_table, ori_cast[['folio','CAST_VAL_ORI']], on=['folio'], how="left")
    
    final_table["CAST_VAL_ORI"] = final_table.apply(lambda x: None if x["MOB"] < x["Mob Castigo"] else x["CAST_VAL_ORI"], axis = 1)
    #FIN NUEVO 2025-11-24: Agregar monto original de castigo

    final_table["Finan"] = final_table.apply(lambda x: x["Monto Fondeado"] if x["MOB"] == 1 else None, axis = 1)
    final_table["CONTEO_FINAN"] = final_table.apply(lambda x: "" if pd.isna(x["Finan"]) else 1, axis = 1)

    #final_table[['BGI2+', 'BGI3+', 'BGI4+', 'BGI5+', 'CAST_VAL']] = final_table[['BGI2+', 'BGI3+', 'BGI4+', 'BGI5+', 'CAST_VAL']].fillna(0)
    final_table[['BGI2+', 'BGI3+', 'BGI4+', 'BGI5+', 'CAST_VAL']] = final_table[['BGI2+', 'BGI3+', 'BGI4+', 'BGI5+', 'CAST_VAL']].replace("", 0)
    final_table["Trimestre"] = final_table.apply(lambda x: "2022-T3" if x["folio"] == "111120220600071" else x["Trimestre"], axis = 1)

    return final_table[final_cols]
    