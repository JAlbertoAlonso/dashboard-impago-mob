import pandas as pd
from datetime import datetime
#from tape_cosechas import create_tape_cosechas
#from tape_cosechas_sin_ajustes import create_tape_cosechas
from tape_cosechas_sin_ajustes_final import create_tape_cosechas
### from tape_cosechas_sin_ajustes_v1_prepagos import create_tape_cosechas
#from tape_cosechas_sin_ajustes_v2_prepagos import create_tape_cosechas
from etl.reports.create_collection_report import create_collection_report
from etl.utils.engines import pg_engine_string, mysql_engine
from etl.utils.secrets_manager import get_secret_datta_db, get_secret_atria_analitica_pg_user


if __name__ == '__main__':
    
    mysql_secret_user = get_secret_datta_db()

    mysql_engine_ = mysql_engine(
        mysql_secret_user.mysql_host,
        mysql_secret_user.mysql_port,
        mysql_secret_user.mysql_user,
        mysql_secret_user.mysql_passwd,
        mysql_secret_user.mysql_db,
    )

    
    
    pgConnection = get_secret_atria_analitica_pg_user()
  
    analitica_db = pg_engine_string(
        pgConnection.POSTGRES_HOST,
        pgConnection.POSTGRES_DB,
        pgConnection.POSTGRES_PORT,
        pgConnection.POSTGRES_USER,
        pgConnection.POSTGRES_PASSWORD,
    )


    #today = pd.to_datetime('today')
    today = pd.to_datetime('2025-12-01 00:00:00')

    #resumen = create_pv_especial(mysql_engine_, analitica_db, '2024-12-01 23:59:59')
    #resumen = create_tape_cosechas(mysql_engine_, analitica_db, today)

    #create_tape_cosechas(mysql_engine_, analitica_db, today)
    
    #Default: Fecha de aplicación = movement_date; Alternativa: Fecha en que se recibio = payment_date
    #create_tape_cosechas(mysql_engine_, analitica_db, today, 'payment_date')
    tape = create_tape_cosechas(mysql_engine_, analitica_db, today, 'payment_date')

    #tape.to_excel('Saldo_cosechas_20251201.xlsx')
    
    #collection_report = create_collection_report(mysql_engine_)
    #collection_report.to_csv('collection_report.csv')
    #print(datetime.now())
    #print("Escribiendo reporte...")
    # 
    #resumen.to_csv('data/pv_complete_{}.csv'.format(today.strftime("%Y-%m-%d %H:%M:%S")), index=False)
    #
    #print(datetime.now())
    #print("FIN")
    
    
    writer = pd.ExcelWriter('Saldo_cosechas_{}.xlsx'.format(today.strftime("%Y-%m-%d %H:%M:%S")), engine='xlsxwriter')
    tape.to_excel(writer, sheet_name='Data', index=False, index_label=None)
    writer.close()
    