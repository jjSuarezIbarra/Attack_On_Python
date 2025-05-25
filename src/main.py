#%% =============================
# Main Script: Predicción de Gastos Recurrentes
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import matplotlib.dates as mdates

# Configurar OMP_NUM_THREADS para evitar memory leak en Windows
os.environ['OMP_NUM_THREADS'] = '1'

# Puedes agregar aquí más imports según se vayan necesitando

#%% =============================
# FASE 2: Carga y Preprocesamiento de Datos
# =============================
clientes = pd.read_csv("../data/clientes.csv")
transacciones = pd.read_csv("../data/transacciones.csv")
transacciones_normal = pd.read_csv("../data/transacciones_normalized.csv")
#%%
def cargar_datos(path_clientes, path_transacciones, nivel='comercio'):
    """
    Carga y preprocesa los datasets de clientes y transacciones.
    Args:
        path_clientes (str): Ruta al archivo clientes.csv
        path_transacciones (str): Ruta al archivo transacciones_normalized.csv
        nivel (str): 'comercio' o 'subgrupo_comercio' para el análisis
    Returns:
        clientes (pd.DataFrame): DataFrame de clientes limpio
        transacciones (pd.DataFrame): DataFrame de transacciones limpio
        merged (pd.DataFrame): DataFrame combinado clientes + transacciones
    """
    # --- Cargar datasets ---
    clientes = pd.read_csv(path_clientes)
    transacciones = pd.read_csv(path_transacciones)

    # --- Conversión de fechas ---
    clientes['fecha_nacimiento'] = pd.to_datetime(clientes['fecha_nacimiento'], errors='coerce')
    clientes['fecha_alta'] = pd.to_datetime(clientes['fecha_alta'], errors='coerce')
    transacciones['fecha'] = pd.to_datetime(transacciones['fecha'], errors='coerce')

    # --- Conversión de montos ---
    transacciones['monto'] = pd.to_numeric(transacciones['monto'], errors='coerce')

    # --- Corregir género faltante para id específico ---
    id_faltante = '91477f382c3cf63ab5cd9263b502109243741158'
    clientes.loc[clientes['id'] == id_faltante, 'genero'] = 'M'

    # --- Selección de nivel de análisis ---
    if nivel not in ['comercio', 'subgrupo_comercio']:
        raise ValueError("El nivel debe ser 'comercio' o 'subgrupo_comercio'")
    if nivel == 'comercio':
        transacciones['nivel_analisis'] = transacciones['comercio']
    else:
        transacciones['nivel_analisis'] = transacciones['subgrupo_comercio']

    # --- Merge de clientes y transacciones ---
    merged = pd.merge(transacciones, clientes, on='id', how='left')

    return clientes, transacciones, merged

#%% =============================
# FASE 3: Detección de Comercios Recurrentes (por cliente-comercio)
# =============================

def calcular_intervalos(transacciones, id_cliente, nombre_comercio, nivel_col='nivel_analisis'):
    """
    Calcula los intervalos (en días) entre transacciones para un cliente y comercio/subgrupo específico.
    Args:
        transacciones (pd.DataFrame): DataFrame de transacciones
        id_cliente (str): ID del cliente
        nombre_comercio (str): Nombre del comercio o subgrupo
        nivel_col (str): Columna a usar como nivel de análisis ('nivel_analisis')
    Returns:
        intervalos (list): Lista de intervalos en días (ordenados por fecha)
        fechas (list): Lista de fechas de las transacciones
    """
    df = transacciones[(transacciones['id'] == id_cliente) & (transacciones[nivel_col] == nombre_comercio)]
    df = df.sort_values('fecha')
    fechas = df['fecha'].tolist()
    intervalos = [(fechas[i] - fechas[i-1]).days for i in range(1, len(fechas))]
    return intervalos, fechas


def analizar_recurrencia(intervalos, porcentaje_aceptacion=0.7, tolerancia=0.2, min_repeticiones=3):
    """
    Analiza si los intervalos presentan recurrencia según criterios de porcentaje, tolerancia y repeticiones.
    Args:
        intervalos (list): Lista de intervalos en días
        porcentaje_aceptacion (float): Proporción mínima de intervalos similares
        tolerancia (float): Tolerancia relativa al promedio (ej. 0.2 para ±20%)
        min_repeticiones (int): Mínimo de repeticiones para considerar recurrencia
    Returns:
        es_recurrente (bool): True si cumple criterios de recurrencia
        info (dict): Información adicional (promedio, std, porcentaje_similares, etc.)
    """
    if len(intervalos) < min_repeticiones:
        return False, {'razon': 'No hay suficientes repeticiones'}
    promedio = np.mean(intervalos)
    std = np.std(intervalos)
    rango_min = promedio * (1 - tolerancia)
    rango_max = promedio * (1 + tolerancia)
    similares = [i for i in intervalos if rango_min <= i <= rango_max]
    porcentaje_similares = len(similares) / len(intervalos) if intervalos else 0
    es_recurrente = porcentaje_similares >= porcentaje_aceptacion
    info = {
        'promedio_intervalo': promedio,
        'desviacion_estandar': std,
        'porcentaje_similares': porcentaje_similares,
        'num_intervalos': len(intervalos),
        'num_similares': len(similares),
        'rango_aceptado': (rango_min, rango_max)
    }
    return es_recurrente, info


def clusterizar_intervalos(intervalos, n_clusters=2, max_tiempo=1.0):
    """
    Aplica clustering (k-means) a los intervalos para detectar patrones múltiples.
    Args:
        intervalos (list): Lista de intervalos en días
        n_clusters (int): Número de clusters a buscar
        max_tiempo (float): Tiempo máximo en segundos para el clustering
    Returns:
        etiquetas (np.array): Etiquetas de cluster para cada intervalo
        info_clusters (dict): Información de cada cluster (promedio, tamaño, etc.)
    """
    if len(intervalos) < n_clusters:
        return np.array([]), {}
    
    # Si hay muy pocos intervalos, no hacer clustering
    if len(intervalos) <= 5:
        return np.array([0] * len(intervalos)), {0: {'promedio': np.mean(intervalos), 'std': np.std(intervalos), 'cantidad': len(intervalos)}}
    
    X = np.array(intervalos).reshape(-1, 1)
    
    # Si hay valores únicos menores que n_clusters, ajustar n_clusters
    n_clusters = min(n_clusters, len(np.unique(X)))
    
    try:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=1,  # Reducir el número de inicializaciones
            max_iter=100  # Limitar el número de iteraciones
        ).fit(X)
        etiquetas = kmeans.labels_
    except Exception as e:
        print(f"Error en clustering: {e}")
        return np.array([0] * len(intervalos)), {0: {'promedio': np.mean(intervalos), 'std': np.std(intervalos), 'cantidad': len(intervalos)}}
    
    info_clusters = {}
    for c in range(n_clusters):
        miembros = X[etiquetas == c].flatten()
        info_clusters[c] = {
            'promedio': np.mean(miembros) if len(miembros) > 0 else None,
            'std': np.std(miembros) if len(miembros) > 0 else None,
            'cantidad': len(miembros)
        }
    return etiquetas, info_clusters


def detectar_comercios_recurrentes(transacciones, nivel_col='nivel_analisis', porcentaje_aceptacion=0.7, tolerancia=0.2, min_repeticiones=3, usar_clustering=False, n_clusters=2):
    """
    Identifica comercios/subgrupos recurrentes para todos los clientes.
    Args:
        transacciones (pd.DataFrame): DataFrame de transacciones
        nivel_col (str): Columna a usar como nivel de análisis
        porcentaje_aceptacion (float): Proporción mínima de intervalos similares
        tolerancia (float): Tolerancia relativa al promedio
        min_repeticiones (int): Mínimo de repeticiones para considerar recurrencia
        usar_clustering (bool): Si aplicar clustering a los intervalos
        n_clusters (int): Número de clusters para clustering
    Returns:
        df_result (pd.DataFrame): DataFrame con pares cliente-comercio recurrentes y su info
    """
    resultados = []
    grupos = list(transacciones.groupby(['id', nivel_col]))
    
    # Filtrar grupos con muy pocas transacciones antes de procesar
    grupos_filtrados = [(id_cliente, nombre_comercio, grupo) 
                       for (id_cliente, nombre_comercio), grupo in grupos 
                       if len(grupo) >= min_repeticiones]
    
    for id_cliente, nombre_comercio, grupo in tqdm(grupos_filtrados, desc="Analizando comercios recurrentes"):
        intervalos, fechas = calcular_intervalos(transacciones, id_cliente, nombre_comercio, nivel_col)
        
        # Si hay muy pocos intervalos, no hacer clustering
        if len(intervalos) < min_repeticiones:
            continue
            
        es_recurrente, info = analizar_recurrencia(intervalos, porcentaje_aceptacion, tolerancia, min_repeticiones)
        cluster_info = None
        
        if usar_clustering and len(intervalos) >= n_clusters:
            etiquetas, info_clusters = clusterizar_intervalos(intervalos, n_clusters)
            cluster_info = info_clusters
            
        resultados.append({
            'id': id_cliente,
            nivel_col: nombre_comercio,
            'es_recurrente': es_recurrente,
            **info,
            'intervalos': intervalos,
            'fechas': fechas,
            'cluster_info': cluster_info
        })
    
    df_result = pd.DataFrame(resultados)
    return df_result

#%% =============================
# FASE 4: Preparación de Series Temporales para Prophet
# =============================

def crear_serie_temporal(transacciones, id_cliente, nombre_comercio, nivel_col='nivel_analisis', col_fecha='fecha', col_monto='monto'):
    """
    Prepara la serie temporal para Prophet para un cliente-comercio/subgrupo.
    Args:
        transacciones (pd.DataFrame): DataFrame de transacciones
        id_cliente (str): ID del cliente
        nombre_comercio (str): Nombre del comercio o subgrupo
        nivel_col (str): Columna a usar como nivel de análisis
        col_fecha (str): Nombre de la columna de fecha
        col_monto (str): Nombre de la columna de monto
    Returns:
        df_prophet (pd.DataFrame): DataFrame con columnas 'ds' (fecha) y 'y' (monto)
    """
    df = transacciones[(transacciones['id'] == id_cliente) & (transacciones[nivel_col] == nombre_comercio)].copy()
    df = df.sort_values(col_fecha)
    df_prophet = df[[col_fecha, col_monto]].rename(columns={col_fecha: 'ds', col_monto: 'y'})
    return df_prophet


def rellenar_fechas_faltantes(df, freq='D', col_fecha='ds', col_monto='y'):
    """
    Rellena fechas faltantes en la serie temporal con monto 0 (SOLO PARA VISUALIZACIÓN, NO PARA PROPHET).
    Args:
        df (pd.DataFrame): DataFrame con columnas de fecha y monto
        freq (str): Frecuencia deseada ('D', 'W', 'M', etc.)
        col_fecha (str): Nombre de la columna de fecha
        col_monto (str): Nombre de la columna de monto
    Returns:
        df_completo (pd.DataFrame): DataFrame con fechas completas y monto 0 donde no hubo transacción
    """
    fechas_completas = pd.date_range(start=df[col_fecha].min(), end=df[col_fecha].max(), freq=freq)
    df_completo = pd.DataFrame({col_fecha: fechas_completas})
    df_completo = df_completo.merge(df, on=col_fecha, how='left').fillna({col_monto: 0})
    return df_completo


def preparar_series_para_prophet(transacciones, recurrentes_df, nivel_col='nivel_analisis', col_fecha='fecha', col_monto='monto'):
    """
    Prepara un diccionario de series temporales para Prophet para todos los pares cliente-comercio recurrentes.
    Args:
        transacciones (pd.DataFrame): DataFrame de transacciones
        recurrentes_df (pd.DataFrame): DataFrame de pares recurrentes
        nivel_col (str): Columna de nivel de análisis
        col_fecha (str): Nombre de la columna de fecha
        col_monto (str): Nombre de la columna de monto
    Returns:
        series_dict (dict): Diccionario {(id, comercio): df_prophet}
    """
    series_dict = {}
    for _, row in recurrentes_df.iterrows():
        id_cliente = row['id']
        nombre_comercio = row[nivel_col]
        df_prophet = crear_serie_temporal(transacciones, id_cliente, nombre_comercio, nivel_col, col_fecha, col_monto)
        series_dict[(id_cliente, nombre_comercio)] = df_prophet
    return series_dict

#%% =============================
# FASE 5: Modelado y Predicción con Prophet
# =============================

def entrenar_y_predecir_prophet(df_prophet, fecha_inicio_pred, fecha_fin_pred, params_prophet=None):
    """
    Entrena un modelo Prophet y predice para un rango de fechas específico.
    Args:
        df_prophet (pd.DataFrame): Serie temporal con columnas 'ds' y 'y'
        fecha_inicio_pred (str o pd.Timestamp): Fecha de inicio de predicción
        fecha_fin_pred (str o pd.Timestamp): Fecha de fin de predicción
        params_prophet (dict): Parámetros opcionales para Prophet
    Returns:
        forecast (pd.DataFrame): DataFrame con predicciones futuras
        modelo (Prophet): Objeto Prophet entrenado
    """
    if params_prophet is None:
        params_prophet = {}
    modelo = Prophet(
        **params_prophet,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    modelo.fit(df_prophet)
    # Genera las fechas a predecir
    future_dates = pd.date_range(start=fecha_inicio_pred, end=fecha_fin_pred, freq='D')
    future = pd.DataFrame({'ds': future_dates})
    forecast = modelo.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    return forecast, modelo


def predecir_proximo_gasto(forecast, umbral_monto=1.0):
    """
    Predice la fecha y monto del próximo gasto recurrente (mayor al umbral).
    Args:
        forecast (pd.DataFrame): DataFrame de predicciones de Prophet
        umbral_monto (float): Monto mínimo para considerar que hay gasto
    Returns:
        fecha_proximo (pd.Timestamp): Fecha del próximo gasto predicho
        monto_proximo (float): Monto predicho
    """
    futuros = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=forecast.shape[0])]
    futuros = futuros[futuros['yhat'] >= umbral_monto]
    if futuros.empty:
        return None, None
    fila = futuros.iloc[0]
    return fila['ds'], fila['yhat']


def evaluar_predicciones(forecast, df_test, col_fecha='ds', col_monto='y', periodo_test=31):
    """
    Evalúa las predicciones de Prophet usando datos reales de test.
    Args:
        forecast (pd.DataFrame): DataFrame de predicciones de Prophet
        df_test (pd.DataFrame): Datos reales de test (enero 2023)
        col_fecha (str): Nombre de la columna de fecha
        col_monto (str): Nombre de la columna de monto
        periodo_test (int): Días del periodo de test
    Returns:
        dict: Métricas de desempeño (MAE, RMSE, etc.)
    """
    # Filtrar predicciones y test al periodo de test
    fecha_inicio = df_test[col_fecha].min()
    fecha_fin = df_test[col_fecha].max()
    pred_test = forecast[(forecast['ds'] >= fecha_inicio) & (forecast['ds'] <= fecha_fin)]
    real_test = df_test[(df_test[col_fecha] >= fecha_inicio) & (df_test[col_fecha] <= fecha_fin)]
    # Alinear por fecha
    merged = pd.merge(pred_test[[col_fecha, 'yhat']], real_test[[col_fecha, col_monto]], on=col_fecha, how='inner')
    if merged.empty:
        return {'MAE': None, 'RMSE': None, 'n': 0}
    mae = np.mean(np.abs(merged['yhat'] - merged[col_monto]))
    rmse = np.sqrt(np.mean((merged['yhat'] - merged[col_monto])**2))
    return {'MAE': mae, 'RMSE': rmse, 'n': len(merged)}


def pipeline_prophet_todos(series_dict, fecha_inicio_pred, fecha_fin_pred, params_prophet=None, umbral_monto=1.0):
    """
    Corre el pipeline de Prophet para todos los pares cliente-comercio recurrentes.
    Args:
        series_dict (dict): Diccionario {(id, comercio): df_prophet} de entrenamiento
        fecha_inicio_pred (str o pd.Timestamp): Fecha de inicio de predicción
        fecha_fin_pred (str o pd.Timestamp): Fecha de fin de predicción
        params_prophet (dict): Parámetros opcionales para Prophet
        umbral_monto (float): Monto mínimo para considerar gasto
    Returns:
        resultados (list): Lista de dicts con resultados por par
    """
    resultados = []
    for key, df_train in tqdm(series_dict.items(), desc="Entrenando modelos Prophet"):
        try:
            forecast, modelo = entrenar_y_predecir_prophet(df_train, fecha_inicio_pred, fecha_fin_pred, params_prophet)
            # No hay test_dict ni evaluación aquí, solo predicción
            resultados.append({
                'id': key[0],
                'comercio': key[1],
                'forecast': forecast
            })
        except Exception as e:
            print(f"Error procesando par {key}: {e}")
            continue
    return resultados

#%% =============================
# FASE 6: Visualización y Reporte
# =============================

def plot_serie_temporal(df, col_fecha='ds', col_monto='y', titulo=None):
    """
    Visualiza la serie temporal de un cliente-comercio.
    Args:
        df (pd.DataFrame): DataFrame con columnas de fecha y monto
        col_fecha (str): Nombre de la columna de fecha
        col_monto (str): Nombre de la columna de monto
        titulo (str): Título de la gráfica
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df[col_fecha], df[col_monto], marker='o')
    plt.xlabel('Fecha')
    plt.ylabel('Monto')
    if titulo:
        plt.title(titulo)
    else:
        plt.title('Serie temporal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_intervalos(intervalos, etiquetas=None, titulo=None):
    """
    Visualiza los intervalos entre transacciones, opcionalmente coloreando por cluster.
    Args:
        intervalos (list): Lista de intervalos en días
        etiquetas (list/np.array): Etiquetas de cluster (opcional)
        titulo (str): Título de la gráfica
    """
    plt.figure(figsize=(10, 4))
    if etiquetas is not None and len(etiquetas) == len(intervalos):
        for c in np.unique(etiquetas):
            idx = np.where(np.array(etiquetas) == c)[0]
            plt.bar(np.array(idx), np.array(intervalos)[idx], label=f'Cluster {c}')
        plt.legend()
    else:
        plt.bar(range(len(intervalos)), intervalos)
    plt.xlabel('Transacción')
    plt.ylabel('Intervalo (días)')
    if titulo:
        plt.title(titulo)
    else:
        plt.title('Intervalos entre transacciones')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predicciones_vs_real(forecast, df_real, col_fecha='ds', col_pred='yhat', col_real='y', titulo=None):
    """
    Visualiza las predicciones de Prophet vs. los datos reales.
    Args:
        forecast (pd.DataFrame): DataFrame de predicciones de Prophet
        df_real (pd.DataFrame): DataFrame de datos reales
        col_fecha (str): Nombre de la columna de fecha
        col_pred (str): Nombre de la columna de predicción
        col_real (str): Nombre de la columna real
        titulo (str): Título de la gráfica
    """
    plt.figure(figsize=(12, 4))
    plt.plot(forecast[col_fecha], forecast[col_pred], label='Predicción (yhat)')
    plt.plot(df_real[col_fecha], df_real[col_real], label='Real', marker='o', linestyle='None')
    plt.xlabel('Fecha')
    plt.ylabel('Monto')
    if titulo:
        plt.title(titulo)
    else:
        plt.title('Predicción vs. Realidad')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_resumen_desempeno(resultados, metrica='MAE', top_n=20):
    """
    Visualiza un resumen de desempeño (ej. MAE) para los pares cliente-comercio.
    Args:
        resultados (list): Lista de dicts con resultados por par
        metrica (str): Métrica a graficar ('MAE', 'RMSE', etc.)
        top_n (int): Número de pares a mostrar
    """
    resumen = []
    for r in resultados:
        if r['metricas'] and r['metricas'][metrica] is not None:
            resumen.append({'id': r['id'], 'comercio': r['comercio'], metrica: r['metricas'][metrica]})
    df_resumen = pd.DataFrame(resumen).sort_values(metrica).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(df_resumen['id'].astype(str) + ' | ' + df_resumen['comercio'].astype(str), df_resumen[metrica])
    plt.xlabel(metrica)
    plt.ylabel('Cliente | Comercio')
    plt.title(f'Top {top_n} pares con menor {metrica}')
    plt.tight_layout()
    plt.show()


def reporte_textual(resultados, top_n=5):
    """
    Genera un reporte textual/resumido de los mejores y peores pares según MAE.
    Args:
        resultados (list): Lista de dicts con resultados por par
        top_n (int): Número de pares a mostrar
    Returns:
        str: Texto resumen
    """
    resumen = []
    for r in resultados:
        if r['metricas'] and r['metricas']['MAE'] is not None:
            resumen.append({'id': r['id'], 'comercio': r['comercio'], 'MAE': r['metricas']['MAE']})
    df_resumen = pd.DataFrame(resumen)
    top_mejores = df_resumen.sort_values('MAE').head(top_n)
    top_peores = df_resumen.sort_values('MAE', ascending=False).head(top_n)
    texto = '--- TOP MEJORES (menor MAE) ---\n'
    texto += top_mejores.to_string(index=False) + '\n\n'
    texto += '--- TOP PEORES (mayor MAE) ---\n'
    texto += top_peores.to_string(index=False) + '\n'
    return texto

def muestrear_datos(transacciones, clientes, porcentaje=0.1, random_state=42):
    """
    Muestra un subconjunto aleatorio de los datos para pruebas rápidas.
    Args:
        transacciones (pd.DataFrame): DataFrame de transacciones
        clientes (pd.DataFrame): DataFrame de clientes
        porcentaje (float): Porcentaje de datos a mantener (0-1)
        random_state (int): Semilla aleatoria para reproducibilidad
    Returns:
        transacciones_muestra (pd.DataFrame): Muestra de transacciones
        clientes_muestra (pd.DataFrame): Muestra de clientes
    """
    # Seleccionar un subconjunto aleatorio de clientes
    clientes_muestra = clientes.sample(frac=porcentaje, random_state=random_state)
    
    # Filtrar transacciones para mantener solo las de los clientes seleccionados
    transacciones_muestra = transacciones[transacciones['id'].isin(clientes_muestra['id'])]
    
    return transacciones_muestra, clientes_muestra

def guardar_predicciones_rango(resultados, fecha_inicio, fecha_fin, df_real=None, output_path=None, clientes=None):
    """
    Guarda las predicciones de un rango de fechas en un CSV y genera gráficos.
    Args:
        resultados (list): Lista de dicts con resultados por par
        fecha_inicio (str o pd.Timestamp): Fecha de inicio del rango
        fecha_fin (str o pd.Timestamp): Fecha de fin del rango
        df_real (pd.DataFrame, opcional): DataFrame con valores reales (id, comercio, fecha, valor_real)
        output_path (str, opcional): Ruta donde guardar el CSV. Si es None, se genera automáticamente.
        clientes (pd.DataFrame, opcional): DataFrame de clientes para agregar género y edad
    """
    # Formatear fechas para el nombre del archivo
    fecha_inicio_str = pd.to_datetime(fecha_inicio).strftime('%Y%m%d')
    fecha_fin_str = pd.to_datetime(fecha_fin).strftime('%Y%m%d')
    if output_path is None:
        output_path = f"../data/predicciones_{fecha_inicio_str}_{fecha_fin_str}.csv"

    # Crear DataFrame con todas las predicciones
    predicciones = []
    for r in resultados:
        try:
            df_pred = r['forecast']
            # Filtrar solo el rango de fechas deseado
            df_pred = df_pred[(df_pred['ds'] >= pd.to_datetime(fecha_inicio)) & (df_pred['ds'] <= pd.to_datetime(fecha_fin))]
            for _, row in df_pred.iterrows():
                predicciones.append({
                    'id': r['id'],
                    'comercio': r['comercio'],
                    'fecha': row['ds'],
                    'valor_predicho': row['yhat']
                })
        except Exception as e:
            print(f"Error procesando par {r['id']}-{r['comercio']}: {e}")
            continue

    if not predicciones:
        print("No se encontraron predicciones válidas para guardar")
        return None

    df_predicciones = pd.DataFrame(predicciones)

    # Si hay valores reales, hacer merge
    if df_real is not None:
        df_real['fecha'] = pd.to_datetime(df_real['fecha'])
        df_predicciones = df_predicciones.merge(df_real, on=['id', 'comercio', 'fecha'], how='left')

    # Agregar género y edad si se pasa el DataFrame de clientes
    if clientes is not None:
        df_predicciones = df_predicciones.merge(
            clientes[['id', 'genero', 'fecha_nacimiento']],
            on='id', how='left'
        )
        df_predicciones['fecha_nacimiento'] = pd.to_datetime(df_predicciones['fecha_nacimiento'], errors='coerce')
        df_predicciones['edad'] = (pd.to_datetime(df_predicciones['fecha']) - df_predicciones['fecha_nacimiento']).dt.days // 365

    df_predicciones.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en: {output_path}")
    print(f"Total de predicciones guardadas: {len(df_predicciones)}")

    # Si tienes valores reales para comparar, calcula error absoluto y métricas
    if 'valor_real' in df_predicciones.columns:
        df_predicciones['error_absoluto'] = abs(df_predicciones['valor_real'] - df_predicciones['valor_predicho'])
        mae_prophet = df_predicciones['error_absoluto'].mean()
        naive_mae = abs(df_predicciones['valor_real'] - df_predicciones['valor_real'].mean()).mean()
        mape = (abs(df_predicciones['valor_real'] - df_predicciones['valor_predicho']) / df_predicciones['valor_real']).mean() * 100 if (df_predicciones['valor_real'] != 0).all() else None
    else:
        mae_prophet = naive_mae = mape = None

    # Gráficos
    if len(df_predicciones) > 0:
        # 1. Gráfico de dispersión: Real vs Predicho (si hay valores reales)
        if 'valor_real' in df_predicciones.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df_predicciones['valor_real'], df_predicciones['valor_predicho'], alpha=0.5)
            plt.plot([0, df_predicciones['valor_real'].max()], [0, df_predicciones['valor_real'].max()], 'r--')
            plt.xlabel('Valor Real')
            plt.ylabel('Valor Predicho')
            plt.title(f'Predicciones vs Valores Reales - {fecha_inicio} a {fecha_fin}')
            texto = f"MAE Prophet: {mae_prophet:.2f}\nMAE naive: {naive_mae:.2f}"
            if mape is not None:
                texto += f"\nMAPE: {mape:.2f}%"
            plt.gca().text(0.05, 0.95, texto, transform=plt.gca().transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            plt.tight_layout()
            plt.show()

        # 2. Gráfico de error absoluto por comercio (si hay valores reales)
        if 'error_absoluto' in df_predicciones.columns:
            plt.figure(figsize=(12, 6))
            error_por_comercio = df_predicciones.groupby('comercio')['error_absoluto'].mean().sort_values(ascending=False)
            error_por_comercio.head(20).plot(kind='bar')
            plt.title(f'Error Absoluto Medio por Comercio - {fecha_inicio} a {fecha_fin}')
            plt.xlabel('Comercio')
            plt.ylabel('Error Absoluto Medio')
            plt.xticks(rotation=45, ha='right')
            plt.gca().text(0.95, 0.95, f"MAE Prophet: {mae_prophet:.2f}\nMAE naive: {naive_mae:.2f}", 
                           transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            plt.tight_layout()
            plt.show()

        # 3. Gráfico de error absoluto por día (si hay valores reales)
        if 'error_absoluto' in df_predicciones.columns:
            plt.figure(figsize=(12, 6))
            error_por_dia = df_predicciones.groupby('fecha')['error_absoluto'].mean()
            error_por_dia.plot(kind='line', marker='o')
            plt.title(f'Error Absoluto Medio por Día - {fecha_inicio} a {fecha_fin}')
            plt.xlabel('Fecha')
            plt.ylabel('Error Absoluto Medio')
            plt.xticks(rotation=45)
            plt.gca().text(0.95, 0.95, f"MAE Prophet: {mae_prophet:.2f}\nMAE naive: {naive_mae:.2f}", 
                           transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            plt.tight_layout()
            plt.show()

    return df_predicciones

#%% =============================
# PIPELINE COMPLETO
# =============================
# 1. Cargar y preprocesar datos
clientes, transacciones, merged = cargar_datos(
    path_clientes="../data/clientes.csv",
    path_transacciones="../data/transacciones_normalized.csv",
    nivel="comercio"  # o 'subgrupo_comercio'
)
#%%
# 1.1 Filtrar solo transacciones de 2022 (excepto enero 2023)
print("Filtrando transacciones de 2022...")
transacciones_2022 = transacciones[
    (transacciones['fecha'] >= '2022-01-01') &
    (transacciones['fecha'] < '2023-01-01')
]
print(f"Transacciones 2022: {len(transacciones_2022)}")
#%%
# 2. Detectar comercios recurrentes en TODO el 2022
print("\nDetectando comercios recurrentes...")
recurrentes_df = detectar_comercios_recurrentes(
    transacciones=transacciones_2022,
    nivel_col="nivel_analisis",
    porcentaje_aceptacion=0.7,
    tolerancia=0.2,
    min_repeticiones=3,
    usar_clustering=True,
    n_clusters=2
)
recurrentes = recurrentes_df[recurrentes_df['es_recurrente']]
print(f"Comercios recurrentes encontrados: {len(recurrentes)}")
#%%
# 3. Muestrear sobre los pares cliente-comercio recurrentes
ptc = 1  # porcentaje de muestra
print(f"\nMuestreando {ptc*100:.0f}% de los pares recurrentes...")
recurrentes_sample = recurrentes.sample(frac=ptc, random_state=42) if ptc < 1.0 else recurrentes
print(f"Pares recurrentes en muestra: {len(recurrentes_sample)}")
#%%
# 4. Filtrar todas las transacciones (2022 y 2023) de los clientes muestreados
ids_sample = recurrentes_sample['id'].unique()
transacciones_todos = transacciones[transacciones['id'].isin(ids_sample)]
#%%
# 5. Preparar series para Prophet SOLO para los pares muestreados
print("\nPreparando series temporales...")
series_dict = preparar_series_para_prophet(
    transacciones=transacciones_todos,
    recurrentes_df=recurrentes_sample,
    nivel_col="nivel_analisis",
    col_fecha="fecha",
    col_monto="monto"
)
print(f"Series temporales preparadas: {len(series_dict)}")
#%%
# 6. Separar train/test (enero 2023 para test)
print("\nSeparando datos de entrenamiento y prueba...")
test_dict = {}
for key, df in series_dict.items():
    # Test: enero 2023
    df_test = df[(df['ds'] >= pd.Timestamp('2023-01-01')) & (df['ds'] <= pd.Timestamp('2023-01-31'))]
    df_train = df[df['ds'] < pd.Timestamp('2023-01-01')]
    series_dict[key] = df_train
    test_dict[key] = df_test
#%%
# 7. Entrenar y predecir con Prophet para todos los pares
print("\nEntrenando modelos y realizando predicciones para varios meses...")

# Entrena solo con datos de 2022
for key, df in series_dict.items():
    df_train = df[(df['ds'] >= pd.Timestamp('2022-01-01')) & (df['ds'] <= pd.Timestamp('2022-12-31'))]
    series_dict[key] = df_train

# Definir los meses a predecir
meses_a_predecir = [
    (pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-31')),
    # (pd.Timestamp('2023-02-01'), pd.Timestamp('2023-02-28')),
    # (pd.Timestamp('2023-06-01'), pd.Timestamp('2023-06-30')),
    # (pd.Timestamp('2023-12-01'), pd.Timestamp('2023-12-31')),
]

for fecha_inicio_pred, fecha_fin_pred in meses_a_predecir:
    print(f"\nPrediciendo para el rango: {fecha_inicio_pred.date()} a {fecha_fin_pred.date()}")
    resultados = pipeline_prophet_todos(
        series_dict=series_dict,
        fecha_inicio_pred=fecha_inicio_pred,
        fecha_fin_pred=fecha_fin_pred,
        params_prophet=None,
        umbral_monto=1.0
    )
    # Guardar predicciones y generar gráficos
    df_predicciones = guardar_predicciones_rango(resultados, fecha_inicio_pred, fecha_fin_pred, df_real=None, output_path=None, clientes=clientes)
    # Mostrar reporte textual de desempeño (solo si hay valores reales)
    if df_predicciones is not None and 'error_absoluto' in df_predicciones.columns:
        print("\nReporte de desempeño:")
        print(reporte_textual(resultados, top_n=5))
    else:
        print("\nNo hay valores reales para calcular el reporte de desempeño en este rango.")
    # Análisis de desempeño global
    if df_predicciones is not None and len(df_predicciones) > 0 and 'valor_real' in df_predicciones.columns and 'error_absoluto' in df_predicciones.columns:
        print("\n--- Análisis de desempeño global ---")
        promedio_real = df_predicciones['valor_real'].mean()
        print(f"Promedio valor_real: {promedio_real:.2f}")
        mae_prophet = df_predicciones['error_absoluto'].mean()
        print(f"MAE Prophet: {mae_prophet:.2f}")
        naive_mae = abs(df_predicciones['valor_real'] - promedio_real).mean()
        print(f"MAE modelo naive (promedio histórico): {naive_mae:.2f}")
        if (df_predicciones['valor_real'] != 0).all():
            mape = (abs(df_predicciones['valor_real'] - df_predicciones['valor_predicho']) / df_predicciones['valor_real']).mean() * 100
            print(f"MAPE Prophet: {mape:.2f}%")
        else:
            print("No se puede calcular MAPE porque hay valores reales igual a 0.")
    else:
        print("\nNo hay valores reales para calcular el análisis de desempeño global en este rango.")

# Al final del pipeline, después de todas las predicciones:
# Calcular y mostrar el porcentaje de clientes cubiertos
if 'clientes' in globals():
    clientes_predichos = set()
    for fecha_inicio_pred, fecha_fin_pred in meses_a_predecir:
        output_path = f"../data/predicciones_{fecha_inicio_pred.strftime('%Y%m%d')}_{fecha_fin_pred.strftime('%Y%m%d')}.csv"
        if os.path.exists(output_path):
            df_pred = pd.read_csv(output_path)
            clientes_predichos.update(df_pred['id'].unique())
    porcentaje_cubierto = 100 * len(clientes_predichos) / len(clientes)
    print(f"\nPorcentaje de clientes cubiertos por las predicciones: {porcentaje_cubierto:.2f}% ({len(clientes_predichos)}/{len(clientes)})")

#%%
