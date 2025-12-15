import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple

# Configuración general de la página
st.set_page_config(layout='wide', page_title='Aplicación de Portafolios – Seminario de Finanzas')

# -----------------------------
# Definición de universos y benchmark
# -----------------------------

# Universo de regiones del mundo
REGIONES = {
    'SPLG': 'SPLG',
    'EWC' : 'EWC',
    'IEUR': 'IEUR',
    'EEM' : 'EEM',
    'EWJ' : 'EWJ'
}

# Universo de sectores de Estados Unidos
SECTORES = {
    'XLC': 'XLC','XLY':'XLY','XLP':'XLP','XLE':'XLE','XLF':'XLF',
    'XLV':'XLV','XLI':'XLI','XLB':'XLB','XLRE':'XLRE','XLK':'XLK','XLU':'XLU'
}

# Pesos del portafolio de mercado (benchmark) para cada universo
BENCHMARK = {
    'Regiones': {
        'SPLG': 0.7062, 'EWC': 0.0323, 'IEUR': 0.1176, 'EEM': 0.0902, 'EWJ': 0.0537
    },
    'Sectores': {
        'XLC':0.0999,'XLY':0.1025,'XLP':0.0482,'XLE':0.0295,'XLF':0.1307,
        'XLV':0.0958,'XLI':0.0809,'XLB':0.0166,'XLRE':0.0187,'XLK':0.3535,'XLU':0.0237
    }
}

# -----------------------------
# Descarga de precios
# -----------------------------

@st.cache_data
def descargar_precios(tickers, fecha_i, fecha_f):
    """
    Descarga precios ajustados de cierre para la lista de tickers y rango de fechas.
    Si el resultado viene con columnas multiíndice, se toma sólo el cierre.
    Elimina columnas sin datos.
    """
    df = yf.download(
        tickers,
        start=fecha_i,
        end=fecha_f,
        auto_adjust=True,
        progress=False,
        threads=False,
        repair=True
    )

    # Si viene con MultiIndex en las columnas, tomar sólo el precio de cierre
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']

    # Quitar tickers sin información
    df = df.dropna(axis=1, how='all')

    return df

# -----------------------------
# Cálculo de métricas
# -----------------------------

def r_historicos(precios: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula rendimientos simples a partir de precios.
    """
    return precios.pct_change().dropna()

def r_portafolio(pesos: np.ndarray, media_retornos: np.ndarray, periodo=250) -> float:
    """
    Calcula el rendimiento anualizado del portafolio.
    """
    return np.dot(pesos, media_retornos) * periodo

def volatilidad_portafolio(pesos: np.ndarray, cov_matriz: np.ndarray, periodo=250) -> float:
    """
    Calcula la volatilidad anualizada del portafolio.
    """
    return np.sqrt(pesos.T @ cov_matriz @ pesos) * np.sqrt(periodo)

def sharpe_ratio(retorno_anual, vol_anual, rf=0.0):
    """
    Calcula el índice de Sharpe.
    """
    return (retorno_anual - rf) / vol_anual if vol_anual != 0 else np.nan

def sortino_ratio(returns: pd.Series, r_objetivo=0.0, periodo=250):
    """
    Calcula el índice de Sortino usando la desviación a la baja.
    """
    rendimiento_medio = returns.mean() * periodo
    retornos_abajo = returns[returns < r_objetivo]
    if retornos_abajo.empty:
        return np.nan
    riesgo_abajo = np.sqrt((retornos_abajo**2).mean()) * np.sqrt(periodo)
    return (rendimiento_medio - r_objetivo) / riesgo_abajo

def max_drawdown(serie: pd.Series):
    """
    Calcula la máxima caída (drawdown máximo) de una serie de riqueza acumulada.
    """
    max_acum = serie.cummax()
    drawdown = serie / max_acum - 1
    return drawdown.min()

def sesgo_curtosis(returns: pd.Series):
    """
    Regresa sesgo y curtosis de una serie de rendimientos.
    """
    return returns.skew(), returns.kurtosis()

def var_hist(returns: pd.Series, alpha=0.05):
    """
    Calcula el Valor en Riesgo (VaR) histórico al nivel alpha.
    """
    return np.quantile(returns, alpha)

def cvar_hist(returns: pd.Series, alpha=0.05):
    """
    Calcula el Valor en Riesgo Condicional (CVaR) histórico al nivel alpha.
    """
    nivel_var = var_hist(returns, alpha)
    return returns[returns <= nivel_var].mean()

# -----------------------------
# Optimizaciones clásicas (Markowitz)
# -----------------------------

def minimizar_volatilidad(cov_matriz: np.ndarray, periodo=250):
    """
    Obtiene el portafolio de mínima varianza (sin ventas en corto).
    """
    n = cov_matriz.shape[0]
    x0 = np.repeat(1/n, n)
    bounds = tuple((0,1) for _ in range(n))
    restricciones = (
        {'type':'eq','fun': lambda x: np.sum(x)-1},
    )

    def funcion_objetivo(x):
        return np.sqrt(x.T @ cov_matriz @ x) * np.sqrt(periodo)

    res = minimize(funcion_objetivo, x0, method='SLSQP', bounds=bounds, constraints=restricciones)
    return res.x

def maximizar_sharpe(media_retornos: np.ndarray, cov_matriz: np.ndarray, rf=0.0, periodo=250):
    """
    Obtiene el portafolio que maximiza el índice de Sharpe (sin ventas en corto).
    """
    n = len(media_retornos)
    x0 = np.repeat(1/n, n)
    bounds = tuple((0,1) for _ in range(n))
    restricciones = (
        {'type':'eq','fun': lambda x: np.sum(x)-1},
    )

    def sharpe_negativo(x):
        r = r_portafolio(x, media_retornos, periodo=periodo)
        v = volatilidad_portafolio(x, cov_matriz, periodo=periodo)
        if v == 0:
            return 1e6
        return - (r - rf) / v

    res = minimize(sharpe_negativo, x0, method='SLSQP', bounds=bounds, constraints=restricciones)
    return res.x

def markowitz_rend(media_retornos: np.ndarray, cov_matriz: np.ndarray, r_objetivo: float, periodo=250):
    """
    Obtiene el portafolio de mínima varianza sujeto a un rendimiento objetivo anual.
    """
    n = len(media_retornos)
    x0 = np.repeat(1/n, n)
    bounds = tuple((0,1) for _ in range(n))

    restricciones = (
        {'type':'eq','fun': lambda x: np.sum(x)-1},
        {'type':'ineq','fun': lambda x: r_portafolio(x, media_retornos, periodo)-r_objetivo}
    )

    def funcion_objetivo(x):
        return np.sqrt(x.T @ cov_matriz @ x) * np.sqrt(periodo)

    res = minimize(funcion_objetivo, x0, method='SLSQP', bounds=bounds, constraints=restricciones)
    if not res.success:
        st.warning('La optimización de Markowitz con rendimiento objetivo no convergió: ' + res.message)
    return res.x

# -----------------------------
# Modelo Black–Litterman
# -----------------------------

def black_litterman(media_retornos: np.ndarray,
                    cov_matriz: np.ndarray,
                    w_mkt: np.ndarray,
                    P: np.ndarray,
                    Q: np.ndarray,
                    tau: float = 0.05,
                    delta: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementación básica del modelo de Black–Litterman.
    Permite múltiples opiniones (views) absolutas o relativas.
    """

    n = len(media_retornos)
    Sigma = cov_matriz

    # Rendimientos de equilibrio (pi) implícitos en el portafolio de mercado
    w_mkt = w_mkt.reshape(n, 1)
    pi = delta * Sigma @ w_mkt  # pi = delta * Sigma * w_mkt

    # Construcción de tau * Sigma y matriz Omega de incertidumbre de las opiniones (views)
    tauSigma = tau * Sigma
    P = np.asarray(P)
    Q = np.asarray(Q).reshape(-1, 1)
    k = P.shape[0]

    Omega = np.zeros((k, k))
    for i in range(k):
        # Varianza de cada opinión: P_i * (tau * Sigma) * P_i'
        Omega[i, i] = P[i:i+1] @ tauSigma @ P[i:i+1].T

    # Fórmula de Black–Litterman para obtener el vector de rendimientos posterior
    tauSigma_inv = np.linalg.pinv(tauSigma)
    Omega_inv = np.linalg.pinv(Omega)

    M = tauSigma_inv + P.T @ Omega_inv @ P
    M_inv = np.linalg.pinv(M)

    mu_bl = M_inv @ (tauSigma_inv @ pi + P.T @ Omega_inv @ Q)

    # En esta versión, la matriz de covarianzas posterior se aproxima por la misma Sigma
    Sigma_bl = Sigma.copy()

    return mu_bl.flatten(), Sigma_bl

# -----------------------------
# Frontera eficiente
# -----------------------------

def frontera_eficiente(media_retornos, cov_matriz, retornos_range=None, puntos=50, periodo=250):
    """
    Calcula una aproximación de la frontera eficiente generando portafolios
    con diferentes rendimientos objetivo.
    """
    if retornos_range is None:
        rend_anuales = media_retornos * periodo
        retornos_range = np.linspace(rend_anuales.min(), rend_anuales.max(), puntos)

    pesos_lista = []
    vols_lista = []

    for r in retornos_range:
        w = markowitz_rend(media_retornos, cov_matriz, r, periodo)
        pesos_lista.append(w)
        vols_lista.append(volatilidad_portafolio(w, cov_matriz, periodo))

    return retornos_range, np.array(vols_lista), np.array(pesos_lista)

# -----------------------------
# Interfaz de la aplicación
# -----------------------------

st.title('Aplicación de Portafolios – Seminario de Finanzas')
st.markdown('**Supuesto:** se asume un año de 250 días hábiles para anualizar rendimientos y volatilidades.')

# Panel lateral: configuración general
st.sidebar.header('Configuración general')
universo = st.sidebar.selectbox('Universo de inversión', ['Regiones','Sectores'])

if universo == 'Regiones':
    tickers = list(REGIONES.keys())
else:
    tickers = list(SECTORES.keys())

# Fechas por defecto
fecha_inicio_def = datetime(2015,1,1)
fecha_fin_def   = datetime.today()
fecha_i = st.sidebar.date_input('Fecha de inicio', value=fecha_inicio_def)
fecha_f = st.sidebar.date_input('Fecha de fin', value=fecha_fin_def)

# Número de periodos en un año
periodo_an = 250

# Descarga de datos
st.info('Descargando datos de precios...')
precios = descargar_precios(tickers, fecha_i, fecha_f)
if precios.isnull().all().all():
    st.error('No se obtuvieron precios para los activos seleccionados.')
    st.stop()

retornos = r_historicos(precios)
media_retornos = retornos.mean().values
cov_matriz = retornos.cov().values

# Visualización básica
col1, col2 = st.columns([2,1])
with col1:
    st.subheader('Series de precios')
    fig_precios = px.line(precios, x=precios.index, y=precios.columns)
    st.plotly_chart(fig_precios, use_container_width=True)

with col2:
    st.subheader('Matriz de correlaciones')
    corr = retornos.corr()
    fig_corr = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------
# 1) Portafolio arbitrario
# -----------------------------

st.header('1) Portafolio arbitrario')
st.markdown('Defina los pesos del portafolio. Puede tomar como referencia el portafolio benchmark.')

# Inicializamos session_state para pesos del usuario
# También re-inicializamos si cambia el universo y el número de tickers
if "pesos_usuario" not in st.session_state or len(st.session_state.pesos_usuario) != len(tickers):
    st.session_state.pesos_usuario = np.array(
        [BENCHMARK[universo].get(t, 1/len(tickers)) for t in tickers],
        dtype=float
    )

# Botón para cargar directamente los pesos del benchmark
if st.button("Usar pesos benchmark"):
    st.session_state.pesos_usuario = np.array(
        [BENCHMARK[universo].get(t, 1/len(tickers)) for t in tickers],
        dtype=float
    )
    st.success("Pesos establecidos iguales al portafolio benchmark.")
    st.rerun()

columnas_pesos = st.columns(len(tickers))

# El usuario modifica pesos → se guardan en session_state
for i, t in enumerate(tickers):
    with columnas_pesos[i]:
        nuevo_valor = st.number_input(
            f'Peso {t}',
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(st.session_state.pesos_usuario[i]),
            key=f"input_{t}"
        )
        st.session_state.pesos_usuario[i] = nuevo_valor

# Tomamos los pesos actualizados
pesos_usuario = st.session_state.pesos_usuario.copy()
total_pesos = pesos_usuario.sum()

# Tolerancia numérica para evitar falsos positivos
epsilon = 1e-6

# Validación: los pesos NO pueden sumar más de 1 (con tolerancia)
if total_pesos > 1 + epsilon:
    st.error(
        f"⚠️ Los pesos suman {total_pesos:.4f}. "
        "La suma NO puede ser mayor a 1. Ajuste los valores."
    )
    st.stop()

st.write(f"**Suma total de pesos:** {total_pesos:.4f}")

# Métricas del portafolio arbitrario
rend_anual = r_portafolio(pesos_usuario, media_retornos, periodo_an)
vol_anual = volatilidad_portafolio(pesos_usuario, cov_matriz, periodo_an)
sharpe_arbitrario = sharpe_ratio(rend_anual, vol_anual)

colA, colB, colC, colD = st.columns(4)
colA.metric('Rendimiento anualizado', f'{rend_anual:.2%}')
colB.metric('Volatilidad anualizada', f'{vol_anual:.2%}')
colC.metric('Índice de Sharpe', f'{sharpe_arbitrario:.3f}')
colD.metric('Pesos', ', '.join([f'{t}:{w:.2%}' for t,w in zip(tickers,pesos_usuario)]))

# Métricas más detalladas
st.subheader('Estadísticos detallados del portafolio arbitrario')
serie_ret_port = (retornos @ pesos_usuario)
serie_riqueza = (1+serie_ret_port).cumprod()

with st.expander('Mostrar métricas completas'):
    skew, kurt = sesgo_curtosis(serie_ret_port)
    md = max_drawdown(serie_riqueza)
    var5 = var_hist(serie_ret_port, 0.05)
    cvar5 = cvar_hist(serie_ret_port, 0.05)
    sortino = sortino_ratio(serie_ret_port, r_objetivo=0.0, periodo=periodo_an)

    st.write({
        'Sesgo': skew,
        'Curtosis': kurt,
        'Máx Drawdown': md,
        'VaR (5%)': var5,
        'CVaR (5%)': cvar5,
        'Índice de Sortino': sortino
    })

    pesos_bench = np.array([BENCHMARK[universo].get(t,0) for t in tickers])
    if pesos_bench.sum() > 0:
        serie_bench = retornos @ pesos_bench
        beta = serie_ret_port.cov(serie_bench)/serie_bench.var()
        st.write({'Beta vs Benchmark': beta})

fig_riqueza = px.line(
    x=serie_riqueza.index,
    y=serie_riqueza.values,
    labels={'x':'Fecha','y':'Índice de riqueza'}
)
st.plotly_chart(fig_riqueza, use_container_width=True)

# -----------------------------
# 2) Portafolios optimizados
# -----------------------------

st.header('2) Portafolios optimizados')
st.markdown('Seleccione las optimizaciones que desea calcular:')

opt_minvar = st.checkbox('Portafolio de mínima varianza', value=True)
opt_sharpe = st.checkbox('Portafolio de máximo Sharpe', value=True)
opt_markowitz = st.checkbox('Portafolio de Markowitz con rendimiento objetivo', value=True)
opt_bl = st.checkbox('Portafolio Black–Litterman (máximo Sharpe)', value=False)

st.markdown("""
**Black–Litterman con múltiples opiniones (views) absolutas**  
En esta implementación, cada view fija un rendimiento esperado anual para un activo específico.
El modelo combina:
- Los rendimientos de equilibrio del mercado (portafolio benchmark),
- Con las opiniones del inversionista,
- De manera consistente con la matriz de covarianzas.
""")

resultados_opt = {}
mu_bl_global = None
Sigma_bl_global = None

# Mínima varianza
if opt_minvar:
    w_minvar = minimizar_volatilidad(cov_matriz, periodo=periodo_an)
    resultados_opt['MinVar'] = w_minvar

# Máximo Sharpe
if opt_sharpe:
    w_sharpe = maximizar_sharpe(media_retornos, cov_matriz, rf=0.0, periodo=periodo_an)
    resultados_opt['MaxSharpe'] = w_sharpe

# Markowitz con rendimiento objetivo
if opt_markowitz:
    st.subheader('Parámetro para Markowitz (rendimiento objetivo clásico)')
    if 'r_objetivo_clasico' not in st.session_state:
        st.session_state.r_objetivo_clasico = float(max(rend_anual, 0.05))

    r_objetivo = st.number_input(
        'Rendimiento objetivo anual',
        min_value=-1.0,
        max_value=5.0,
        value=st.session_state.r_objetivo_clasico,
        step=0.01,
        format="%.4f",
        key='r_objetivo_clasico_input'
    )

    st.session_state.r_objetivo_clasico = r_objetivo

    w_mark = markowitz_rend(media_retornos, cov_matriz, r_objetivo, periodo=periodo_an)
    resultados_opt['Markowitz'] = w_mark

# Black–Litterman con múltiples views absolutas
if opt_bl:
    st.subheader('Parámetros para Black–Litterman (múltiples opiniones)')

    # Número de opiniones (views) que se quieren incorporar
    num_views = st.number_input(
        'Número de opiniones (views) a considerar',
        min_value=1,
        max_value=len(tickers),
        value=1,
        step=1,
        key='num_views_bl'
    )

    # Parámetros globales de Black–Litterman
    tau_bl = st.number_input(
        'Tau (incertidumbre del portafolio de mercado)',
        min_value=0.0001,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.4f",
        key='tau_bl'
    )

    delta_bl = st.number_input(
        'Delta (aversión al riesgo del inversionista)',
        min_value=0.1,
        max_value=10.0,
        value=2.5,
        step=0.1,
        format="%.2f",
        key='delta_bl'
    )

    # Pesos del portafolio de mercado (benchmark) para construir los rendimientos de equilibrio
    w_mkt = np.array([BENCHMARK[universo].get(t, 0.0) for t in tickers])
    if w_mkt.sum() > 0:
        w_mkt = w_mkt / w_mkt.sum()
    else:
        # Si por alguna razón no hay benchmark definido, se usa equiponderado
        w_mkt = np.repeat(1/len(tickers), len(tickers))

    n_activos = len(tickers)

    # Matriz P (num_views x n_activos) y vector Q (num_views)
    P = np.zeros((num_views, n_activos))
    Q_diario = np.zeros(num_views)

    st.markdown('---')
    st.markdown('### Definición de opiniones (views) absolutas')
    st.markdown('Cada opinión fija un rendimiento esperado anual para un activo específico.')

    # Definimos cada view
    for i in range(num_views):
        st.markdown(f'**Opinión (view) {i+1}**')
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            activo_view = st.selectbox(
                f'Activo para la opinión {i+1}',
                tickers,
                key=f'activo_view_{i}'
            )

        with col_v2:
            ret_view_anual = st.number_input(
                f'Rendimiento esperado anual para la opinión {i+1} (por ejemplo 0.10 = 10%)',
                min_value=-1.0,
                max_value=5.0,
                value=0.10,
                step=0.01,
                format="%.4f",
                key=f'ret_view_{i}'
            )

        indice_activo = tickers.index(activo_view)
        P[i, indice_activo] = 1.0
        Q_diario[i] = ret_view_anual / periodo_an

    # Aplicamos el modelo de Black–Litterman con todas las views
    mu_bl, Sigma_bl = black_litterman(
        media_retornos,
        cov_matriz,
        w_mkt,
        P,
        Q_diario,
        tau=tau_bl,
        delta=delta_bl
    )

    mu_bl_global = mu_bl
    Sigma_bl_global = Sigma_bl

    # Portafolio de máximo Sharpe con los rendimientos ajustados por BL
    w_bl = maximizar_sharpe(mu_bl, Sigma_bl, rf=0.0, periodo=periodo_an)
    resultados_opt['BL_MaxSharpe'] = w_bl

    with st.expander('Comparar rendimientos esperados históricos vs Black–Litterman (por periodo)'):
        st.write('Rendimientos esperados históricos (media_retornos):')
        st.write(pd.Series(media_retornos, index=tickers))
        st.write('Rendimientos esperados ajustados por Black–Litterman (mu_bl):')
        st.write(pd.Series(mu_bl, index=tickers))

# -----------------------------
# Resultados de las optimizaciones
# -----------------------------

if resultados_opt:
    st.subheader('Pesos de los portafolios optimizados')

    for nombre, w in resultados_opt.items():
        st.write(f'**{nombre}**: ' + ', '.join([f'{t}:{val:.2%}' for t,val in zip(tickers,w)]))

    st.subheader('Métricas detalladas de los portafolios optimizados')

    tabla_metricas = []

    pesos_bench = np.array([BENCHMARK[universo].get(t,0) for t in tickers])
    serie_bench = None
    if pesos_bench.sum() > 0:
        serie_bench = retornos @ pesos_bench

    for nombre, w in resultados_opt.items():
        serie_ret = retornos @ w
        serie_riqueza_opt = (1 + serie_ret).cumprod()

        r = r_portafolio(w, media_retornos, periodo_an)
        v = volatilidad_portafolio(w, cov_matriz, periodo_an)
        s = sharpe_ratio(r, v)
        sortino = sortino_ratio(serie_ret, r_objetivo=0.0, periodo=periodo_an)
        md = max_drawdown(serie_riqueza_opt)
        skew, kurt = sesgo_curtosis(serie_ret)
        var5 = var_hist(serie_ret, 0.05)
        cvar5 = cvar_hist(serie_ret, 0.05)

        beta = np.nan
        if serie_bench is not None:
            beta = serie_ret.cov(serie_bench) / serie_bench.var()

        tabla_metricas.append({
            'Portafolio': nombre,
            'Rend_Anual': r,
            'Vol_Anual': v,
            'Sharpe': s,
            'Sortino': sortino,
            'Max_Drawdown': md,
            'Sesgo': skew,
            'Curtosis': kurt,
            'VaR_5%': var5,
            'CVaR_5%': cvar5,
            'Beta_vs_Bench': beta
        })

    df_metricas = pd.DataFrame(tabla_metricas).set_index('Portafolio')
    st.dataframe(df_metricas)

    st.subheader('Frontera eficiente aproximada')
    retornos_range, vols, pesos = frontera_eficiente(media_retornos, cov_matriz, puntos=40, periodo=periodo_an)
    fig_front = go.Figure()
    fig_front.add_trace(go.Scatter(
        x=vols,
        y=retornos_range,
        mode='lines',
        name='Frontera clásica (histórica)'
    ))

    if mu_bl_global is not None and Sigma_bl_global is not None:
        retornos_bl, vols_bl, _ = frontera_eficiente(mu_bl_global, Sigma_bl_global, puntos=40, periodo=periodo_an)
        fig_front.add_trace(go.Scatter(
            x=vols_bl,
            y=retornos_bl,
            mode='lines',
            name='Frontera Black–Litterman',
            line=dict(dash='dash')
        ))

    for nombre,w in resultados_opt.items():
        r = r_portafolio(w, media_retornos, periodo_an)
        v = volatilidad_portafolio(w, cov_matriz, periodo_an)
        fig_front.add_trace(go.Scatter(x=[v], y=[r], mode='markers', name=nombre, marker=dict(size=10)))

    fig_front.update_layout(xaxis_title='Volatilidad anualizada', yaxis_title='Rendimiento anualizado')
    st.plotly_chart(fig_front, use_container_width=True)

# -----------------------------
# Exportación de resultados
# -----------------------------

st.header('Exportar resultados')
if resultados_opt:
    df_export = pd.DataFrame({nombre: w for nombre,w in resultados_opt.items()}, index=tickers)
    csv = df_export.to_csv().encode('utf-8')
    st.download_button(
        'Descargar pesos de portafolios (CSV)',
        data=csv,
        file_name='pesos_portafolios_optimizados.csv',
        mime='text/csv'
    )

st.markdown('---')
