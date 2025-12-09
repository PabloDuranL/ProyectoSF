import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple

st.set_page_config(layout='wide', page_title='Portfolio Quant App')

# -----------------------------
# Utilities / Data
# -----------------------------
REGIONES = {
    'SPLG': 'SPLG',
    'EWC' : 'EWC',
    'IEUR': 'IEUR',
    'EEM' : 'EEM',
    'EWJ' : 'EWJ'
}

SECTORES = {
    'XLC': 'XLC','XLY':'XLY','XLP':'XLP','XLE':'XLE','XLF':'XLF',
    'XLV':'XLV','XLI':'XLI','XLB':'XLB','XLRE':'XLRE','XLK':'XLK','XLU':'XLU'
}

BENCHMARK = {
    'Regiones': {
        'SPLG': 0.7062, 'EWC': 0.0323, 'IEUR': 0.1176, 'EEM': 0.0902, 'EWJ': 0.0537
    },
    'Sectores': {
        'XLC':0.0999,'XLY':0.1025,'XLP':0.0482,'XLE':0.0295,'XLF':0.1307,
        'XLV':0.0958,'XLI':0.0809,'XLB':0.0166,'XLRE':0.0187,'XLK':0.3535,'XLU':0.0237
    }
}

@st.cache_data

def descargar_precios(tickers, fecha_i, fecha_f):
    df = yf.download(
        tickers,
        start=fecha_i,
        end=fecha_f,
        auto_adjust=True,
        progress=False,
        threads=False,
        repair=True
    )

    # Si viene MultiIndex → solo tomar el Close ajustado
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']

    # Eliminar tickers sin datos
    df = df.dropna(axis=1, how='all')

    return df

# -----------------------------
# Metricas
# -----------------------------

# Retornos
def r_historicos(precios: pd.DataFrame) -> pd.DataFrame:
    return precios.pct_change().dropna()

# Media
def r_portafolio(pesos: np.ndarray, media_retornos: np.ndarray, periodo=252) -> float:
    return np.dot(pesos, media_retornos) * periodo

# Volatilidad
def volatilidad_portafolio(pesos: np.ndarray, cov_matriz: np.ndarray, periodo=252) -> float:
    return np.sqrt(pesos.T @ cov_matriz @ pesos) * np.sqrt(periodo)

# Sharpe Ratio
def sharpe_ratio(retorno_anual, vol_anual, rf=0.0):
    return (retorno_anual - rf) / vol_anual if vol_anual != 0 else np.nan

# Sortino_Ratio
def sortino_ratio(retornos: pd.Series, r_objetivo=0.0, rf=0.0, periodo=252):
    # r_objetivo: retorno objetivo anual
    downside = retornos[retornos < r_objetivo/periodo]
    if len(downside) == 0:
        return np.nan
    downside_std = downside.std() * np.sqrt(periodo)
    r_anual = retornos.mean() * periodo
    return (r_anual - rf) / downside_std if downside_std != 0 else np.nan


def max_drawdown(retornos_acum: pd.Series) -> float:
    # retornos_acum es una serie del crecimiento acumulado del portafolio en el tiempo
    roll_max = retornos_acum.cummax()
    drawdown = (retornos_acum - roll_max) / roll_max
    return drawdown.min()


def var_hist(retornos: pd.Series, alpha=0.05) -> float:
    return -np.percentile(retornos, alpha*100)


def cvar_hist(retornos: pd.Series, alpha=0.05) -> float:
    var_level = np.percentile(retornos, alpha*100)
    return -retornos[retornos <= var_level].mean()


def sesgo_curtosis(retornos: pd.Series) -> Tuple[float,float]:
    return retornos.skew(), retornos.kurtosis()


def beta_to_benchmark(asset_retornos: pd.Series, benchmark_retornos: pd.Series) -> float:
    cov = asset_retornos.cov(benchmark_retornos)
    var = benchmark_retornos.var()
    return cov/var if var != 0 else np.nan

# -----------------------------
# Optimizacion
# -----------------------------

def minimizar_volatilidad(cov_matriz: np.ndarray, periodo=252):
    n = cov_matriz.shape[0]
    x0 = np.repeat(1/n, n) # Condición inicial: pesos iguales
    bounds = tuple((0,1) for _ in range(n)) # Pesos entre 0 y 1 (no short)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x)-1}) # Suma de pesos = 1
    def fun(x): # Funcion a minimizar
        return np.sqrt(x.T @ cov_matriz @ x) * np.sqrt(periodo)
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x



def maximizar_sharpe(media_retornos: np.ndarray, cov_matriz: np.ndarray, rf=0.0, periodo=252):
    n = len(media_retornos)
    x0 = np.repeat(1/n, n)
    bounds = tuple((0,1) for _ in range(n))
    constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1})
    # Minizar el Sharpe Negativo
    def neg_sharpe(x):
        ret = r_portafolio(x, media_retornos, periodo)
        vol = volatilidad_portafolio(x, cov_matriz, periodo)
        return - (ret - rf)/vol if vol!=0 else 1e6
    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x


def markowitz_rend(media_retornos: np.ndarray, cov_matriz: np.ndarray, r_objetivo: float, periodo=252):
    n = len(media_retornos)
    x0 = np.repeat(1/n, n)
    bounds = tuple((0,1) for _ in range(n))
    # constraint: suma de pesos igual a 1 y retorno del portafolio >= r_objetivo
    cons = (
        {'type':'eq','fun': lambda x: np.sum(x)-1},
        {'type':'ineq','fun': lambda x: r_portafolio(x, media_retornos, periodo)-r_objetivo}
    )
    # Volatilidad(riesgo)
    def fun(x):
        return np.sqrt(x.T @ cov_matriz @ x) * np.sqrt(periodo)
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        st.warning('Optimización por varianza con r_objetivo no convergió: ' + res.message)
    return res.x

# Frontera Eficiente
def frontera_eficiente(media_retornos, cov_matriz, retornos_range=None, points=50, periodo=252): ####################################################
    if retornos_range is None:
        # Si no se proporciona un rango de retornos se toma entre el min y max del portafolio
        an_retornos = media_retornos * periodo
        retornos_range = np.linspace(an_retornos.min(), an_retornos.max(), points)
    pesos = []
    vols = []
    for r in retornos_range:
        w = markowitz_rend(media_retornos, cov_matriz, r, periodo)
        pesos.append(w)
        vols.append(volatilidad_portafolio(w, cov_matriz, periodo))
    return retornos_range, np.array(vols), np.array(pesos)

# -----------------------------
# Streamlit App Layout
# -----------------------------

st.title('Gestión y Análisis de Portafolios')
st.markdown('''
Esta aplicación permite analizar portafolios sobre dos universos (Regiones y Sectores EE. UU.),
calcular métricas de riesgo/retorno y resolver optimizaciones: **Mínima Varianza**, **Máximo Sharpe** y **Markowitz** (con retorno objetivo).
''')

# Sidebar: selección
st.sidebar.header('Configuración de Análisis')
universe = st.sidebar.selectbox('Selecciona universo', options=['Regiones','Sectores'])

if universe == 'Regiones':
    tickers = list(REGIONES.keys())
else:
    tickers = list(SECTORES.keys())

fecha_i = st.sidebar.date_input('Fecha inicio', value=datetime(2010,1,1))  # Fecha de inicio
fecha_f = datetime.today()      # Fecha de fin (hoy)
periodo_an = 252  

st.sidebar.markdown('---')
st.sidebar.subheader('Benchmarks')
st.sidebar.write(BENCHMARK[universe])

# fetch data
st.info('Descargando datos...')
precios = descargar_precios(tickers, fecha_i, fecha_f)
if precios.isnull().all().all():
    st.error('No se obtuvieron precios: verifica los tickers')
    st.stop()

retornos = r_historicos(precios)
media_retornos = retornos.mean().values
cov_matriz = retornos.cov().values

# Show basic visuals
col1, col2 = st.columns([2,1])
with col1:
    st.subheader('Series de precios (Adj Close)')
    fig = px.line(precios, x=precios.index, y=precios.columns)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader('Matriz de correlación')
    corr = retornos.corr()
    fig2 = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Portafolio Arbitrario
# -----------------------------
st.header('1) Portafolio Arbitrario')
st.markdown('Define los pesos para un portafolio arbitrario. Deben sumar 1. Puedes usar los pesos benchmark como referencia.')

default_weights = [BENCHMARK[universe].get(t, 1/len(tickers)) for t in tickers]
user_weights = []
cols = st.columns(len(tickers))
for i, t in enumerate(tickers):
    with cols[i]:
        w = st.number_input(f'{t} peso', value=float(round(default_weights[i],4)), min_value=0.0, max_value=1.0, step=0.0001, key=f'w_{t}')
        user_weights.append(w)
user_weights = np.array(user_weights)

if abs(user_weights.sum()-1) > 1e-6:
    st.warning(f'Los pesos suman {user_weights.sum():.6f}. Deben sumar 1. Puedes normalizarlos manualmente o usar el botón de normalizar.')
    if st.button('Normalizar pesos'):
        user_weights = user_weights / user_weights.sum()
        st.experimental_rerun()

# metricas portafolio arbitrario
r_anual = r_portafolio(user_weights, media_retornos, periodo_an)
ann_vol = volatilidad_portafolio(user_weights, cov_matriz, periodo_an)
sharpe = sharpe_ratio(r_anual, ann_vol)

colA, colB, colC, colD = st.columns(4)
colA.metric('Rend. anualizado', f'{r_anual:.2%}')
colB.metric('Vol. anualizada', f'{ann_vol:.2%}')
colC.metric('Sharpe', f'{sharpe:.3f}')
colD.metric('Pesos (sample)', ', '.join([f'{t}:{w:.2%}' for t,w in zip(tickers,user_weights)]))

# Estadisticas detalladas
st.subheader('Estadísticos detallados (Arbitrario)')
port_ret_series = (retornos @ user_weights)
retornos_acum = (1+port_ret_series).cumprod()

with st.expander('Ver métricas completas'):
    skew, kurt = sesgo_curtosis(port_ret_series)
    md = max_drawdown(retornos_acum)
    var5 = var_hist(port_ret_series, 0.05)
    cvar5 = cvar_hist(port_ret_series, 0.05)
    sortino = sortino_ratio(port_ret_series, r_objetivo=0.0, periodo=periodo_an)

    st.write({'Skewness':skew, 'Kurtosis':kurt, 'Max Drawdown':md, 'VaR(5%)':var5, 'CVaR(5%)':cvar5, 'Sortino':sortino})

    # Beta vs benchmark (portfolio-level) if benchmark series available
    benchmark_w = np.array([BENCHMARK[universe].get(t,0) for t in tickers])
    if benchmark_w.sum() > 0:
        bench_series = retornos @ benchmark_w
        beta = port_ret_series.cov(bench_series)/bench_series.var()
        st.write({'Beta vs Benchmark': beta})

# plot cumulative
fig_cum = px.line(x=retornos_acum.index, y=retornos_acum.values, labels={'x':'Date','y':'Wealth Index'})
st.plotly_chart(fig_cum, use_container_width=True)

# -----------------------------
# Optimizacion
# -----------------------------
st.header('2) Portafolios Optimizados')
st.markdown('Seleccione optimizaciones a ejecutar:')
opt_minvar = st.checkbox('Optimizar: Mínima Varianza', value=True)
opt_sharpe = st.checkbox('Optimizar: Máximo Sharpe', value=True)
opt_markowitz = st.checkbox('Optimizar: Markowitz con retorno objetivo', value=True)

results = {}

if opt_minvar:
    w_minvar = minimizar_volatilidad(cov_matriz)
    results['MinVar'] = w_minvar

if opt_sharpe:
    w_sharpe = maximizar_sharpe(media_retornos, cov_matriz, rf=0.0, periodo=periodo_an)
    results['MaxSharpe'] = w_sharpe

if opt_markowitz:
    st.markdown('**Retorno objetivo anual (decimal)** para Markowitz clásico (ej. 0.08 = 8%):')

    # Usamos session_state para que el valor que elija el usuario no se resetee
    if 'r_objetivo_clasico' not in st.session_state:
        st.session_state.r_objetivo_clasico = float(max(r_anual, 0.05))

    r_objetivo = st.number_input(
        'Retorno objetivo anual clásico',
        min_value=-1.0,
        max_value=5.0,
        value=st.session_state.r_objetivo_clasico,
        step=0.01,
        format="%.4f",
        key='r_objetivo_clasico_input'
    )

    # Actualizamos el valor almacenado con lo que el usuario haya elegido
    st.session_state.r_objetivo_clasico = r_objetivo

    w_mark = markowitz_rend(media_retornos, cov_matriz, r_objetivo, periodo=periodo_an)
    results['Markowitz'] = w_mark

# show results
if results:
    st.subheader('Pesos resultantes')
    for name, w in results.items():
        st.write(f'**{name}**: ' + ', '.join([f'{t}:{val:.2%}' for t,val in zip(tickers,w)]))

    st.subheader('Comparativa métricas')
    table = []
    for name, w in results.items():
        r = r_portafolio(w, media_retornos, periodo_an)
        v = volatilidad_portafolio(w, cov_matriz, periodo_an)
        s = sharpe_ratio(r, v)
        table.append({'Portfolio':name, 'Ann Return':r, 'Ann Vol':v, 'Sharpe':s})
    df_table = pd.DataFrame(table).set_index('Portfolio')
    st.data_editor(df_table)

    # Efficient frontier
    if 'Markowitz' in results or opt_minvar or opt_sharpe:
        st.subheader('Frontera Eficiente (approx)')
        retornos_range, vols, pesos = frontera_eficiente(media_retornos, cov_matriz, points=40, periodo=periodo_an)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vols, y=retornos_range, mode='lines', name='Frontera'))
        # mark points
        for name,w in results.items():
            r = r_portafolio(w, media_retornos, periodo_an)
            v = volatilidad_portafolio(w, cov_matriz, periodo_an)
            fig.add_trace(go.Scatter(x=[v], y=[r], mode='markers', name=name, marker=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Export / Download
# -----------------------------
st.header('Exportar resultados')
if results:
    export_df = pd.DataFrame({name: w for name,w in results.items()}, index=tickers)
    csv = export_df.to_csv().encode('utf-8')
    st.download_button('Descargar pesos (CSV)', data=csv, file_name='optimized_weights.csv', mime='text/csv')

st.markdown('---')
