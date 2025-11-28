import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from datetime import datetime

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
st.set_page_config(page_title="Análisis Integral de Acciones", layout="wide", initial_sidebar_state="collapsed")

# Verificar disponibilidad de Gemini
GEMINI_DISPONIBLE = True
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    modelo_test = genai.GenerativeModel("gemini-2.5-flash")
except:
    GEMINI_DISPONIBLE = False

# Inicializar session_state
if "ticker" not in st.session_state:
    st.session_state["ticker"] = None
if "analizar" not in st.session_state:
    st.session_state["analizar"] = False

# -----------------------------
# FUNCIONES
# -----------------------------
@st.cache_data(ttl=86400)
def obtener_tasa_libre_riesgo():
    """Obtiene la tasa CETES 28 desde Banxico como proxy de tasa libre de riesgo."""
    try:
        url = "https://www.banxico.org.mx/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Buscar el div de CETES 28
        cetes_div = soup.find("div", {"id": "dv_singleCetes28"})
        if cetes_div:
            valor_span = cetes_div.find("span", class_="valor")
            if valor_span:
                tasa = float(valor_span.get_text(strip=True))
                return tasa / 100  # Convertir a decimal
        
        return 0.07  # Default 7% si no se puede obtener
    except:
        return 0.07


@st.cache_data(ttl=3600)
def buscar_empresas_detallado(nombre):
    """Busca empresas en Yahoo Finance y devuelve nombre, ticker, precio, país y logo."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {"q": nombre, "quotes_count": 8, "news_count": 0}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()

        resultados = []
        for item in data.get("quotes", []):
            ticker = item.get("symbol")
            nombre = item.get("shortname") or item.get("longname")
            if not ticker or not nombre:
                continue

            try:
                info = yf.Ticker(ticker).info
                precio = info.get("currentPrice", "N/A")
                pais = info.get("country", "N/A")
                logo = info.get("logo_url", None)
            except:
                precio, pais, logo = "N/A", "N/A", None

            resultados.append({
                "ticker": ticker,
                "nombre": nombre,
                "precio": precio,
                "pais": pais,
                "logo": logo
            })

        return resultados
    except:
        return []


@st.cache_data(ttl=86400)
def obtener_peers_finviz(ticker):
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        peers_div = soup.find("div", class_="flex-1 max-w-max truncate")
        if not peers_div:
            return []
        links = peers_div.find_all("a", class_="tab-link")
        return list(dict.fromkeys([a.get_text(strip=True) for a in links if a.get_text(strip=True).isupper()]))
    except:
        return []


@st.cache_data(ttl=86400)
def obtener_financial_insights_yf(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        insights_div = soup.find("div", {"class": "summary yf-1lj2fxg"})
        if insights_div:
            return insights_div.get_text(strip=True)

        h2_financials = soup.find("h2", string=lambda x: x and "Financials" in x)
        if h2_financials:
            parent = h2_financials.find_parent()
            if parent:
                summary = parent.find("p")
                if summary:
                    return summary.get_text(strip=True)

        return None
    except:
        return None


@st.cache_data(ttl=86400)
def obtener_financial_insights_peers(peers_tickers):
    insights_dict = {}
    for t in peers_tickers:
        insight = obtener_financial_insights_yf(t)
        if insight:
            insights_dict[t] = insight
    return insights_dict


@st.cache_data(ttl=86400)
def obtener_income_yahoo(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.financials
        claves = ["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income", "Net Income", "EBIT", "EBITDA"]
        if df.empty:
            return pd.DataFrame()
        # Filtrar solo claves que existen en el índice
        claves_existentes = [c for c in claves if c in df.index]
        df = df.loc[claves_existentes].reindex(claves).fillna(0).astype(float).T
        df.index = [i.strftime("%Y-%m-%d") for i in df.index]
        data = []
        for m in df.columns:
            vals = [
                f"{v/1e9:,.1f} B" if abs(v) >= 1e9 else
                f"{v/1e6:,.0f} M" if abs(v) >= 1e6 else
                f"{v:,.0f}"
                for v in df[m]
            ]
            data.append([m] + vals)
        return pd.DataFrame(data, columns=["Métrica"] + list(df.index))
    except:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def obtener_balance_yahoo(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.balance_sheet
        claves = ["Total Assets", "Total Liabilities Net Minority Interest", "Total Equity Gross Minority Interest", "Total Debt", "Net Debt"]
        if df.empty:
            return pd.DataFrame()
        claves_existentes = [c for c in claves if c in df.index]
        df = df.loc[claves_existentes].reindex(claves).fillna(0).astype(float).T
        df.index = [i.strftime("%Y-%m-%d") for i in df.index]
        data = []
        for m in df.columns:
            vals = [
                f"{v/1e9:,.1f} B" if abs(v) >= 1e9 else
                f"{v/1e6:,.0f} M" if abs(v) >= 1e6 else
                f"{v:,.0f}"
                for v in df[m]
            ]
            data.append([m] + vals)
        return pd.DataFrame(data, columns=["Métrica"] + list(df.index))
    except:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def obtener_cashflow_yahoo(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.cashflow
        claves = ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Free Cash Flow"]
        if df.empty:
            return pd.DataFrame()
        claves_existentes = [c for c in claves if c in df.index]
        df = df.loc[claves_existentes].reindex(claves).fillna(0).astype(float).T
        df.index = [i.strftime("%Y-%m-%d") for i in df.index]
        data = []
        for m in df.columns:
            vals = [
                f"{v/1e9:,.1f} B" if abs(v) >= 1e9 else
                f"{v/1e6:,.0f} M" if abs(v) >= 1e6 else
                f"{v:,.0f}"
                for v in df[m]
            ]
            data.append([m] + vals)
        return pd.DataFrame(data, columns=["Métrica"] + list(df.index))
    except:
        return pd.DataFrame()


def traducir_descripcion(texto, idioma_destino):
    """Traduce texto solo si Gemini está disponible."""
    if not GEMINI_DISPONIBLE:
        return texto
    
    if idioma_destino == "Inglés" or not texto or texto == "Descripción no disponible.":
        return texto
    try:
        modelo = genai.GenerativeModel("gemini-2.5-flash")
        respuesta = modelo.generate_content(f"Traduce al {idioma_destino}: {texto}")
        return respuesta.text.strip()
    except:
        return texto


@st.cache_data(ttl=86400)
def obtener_kpis_peers(ticker, peers_list):
    resultados = []
    for t in [ticker] + peers_list[:5]:
        try:
            info = yf.Ticker(t).info
            resultados.append({
                "Ticker": t,
                "Nombre": info.get('longName', info.get('shortName', t)),
                "Market Cap (B)": f"{info.get('marketCap', 0)/1e9:.2f}" if info.get('marketCap') else "N/A",
                "P/E": f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A",
                "EPS": f"{info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else "N/A",
                "Beta": f"{info.get('beta', 0):.2f}" if info.get('beta') else "N/A",
                "ROE (%)": f"{info.get('returnOnEquity', 0)*100:.2f}" if info.get('returnOnEquity') else "N/A",
                "Gross Margin (%)": f"{info.get('grossMargins', 0)*100:.2f}" if info.get('grossMargins') else "N/A",
                "Profit Margin (%)": f"{info.get('profitMargins', 0)*100:.2f}" if info.get('profitMargins') else "N/A",
                "Debt/Equity": f"{info.get('debtToEquity', 0)/100:.2f}" if info.get('debtToEquity') else "N/A",
                "Dividend Yield (%)": f"{info.get('dividendYield', 0)*100:.2f}" if info.get('dividendYield') else "0.00",
                "Revenue TTM (B)": f"{info.get('totalRevenue', 0)/1e9:.2f}" if info.get('totalRevenue') else "N/A"
            })
        except:
            continue
    return pd.DataFrame(resultados)


def extraer_precios_columna(datos):
    """Extrae la columna de precios correctamente sin importar si es MultiIndex o no."""
    if datos.empty:
        return pd.Series(dtype=float)
    
    if isinstance(datos.columns, pd.MultiIndex):
        if "Adj Close" in datos.columns.get_level_values(0):
            col = datos["Adj Close"]
        else:
            col = datos["Close"]
        
        if isinstance(col, pd.DataFrame):
            return col.iloc[:, 0].astype(float)
        else:
            return col.astype(float)
    else:
        if "Adj Close" in datos.columns:
            return datos["Adj Close"].astype(float)
        else:
            return datos["Close"].astype(float)


@st.cache_data(ttl=3600)
def descargar_datos_historicos(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False)


@st.cache_data(ttl=3600)
def obtener_noticias_yf(ticker):
    """Obtiene noticias recientes de Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        news = t.news
        return news if news else []
    except:
        return []


def analizar_sentimiento_gemini(ticker, noticias):
    """Analiza el sentimiento de las noticias usando Gemini."""
    if not GEMINI_DISPONIBLE or not noticias:
        return None
    
    headlines = [n.get('title', '') for n in noticias[:10]]
    headlines_text = "\n".join([f"- {h}" for h in headlines])
    
    prompt = f"""
    Analiza el sentimiento de mercado para {ticker} basado en estos titulares recientes:
    {headlines_text}
    
    Clasifica el sentimiento general como: POSITIVO, NEUTRAL o NEGATIVO.
    Provee un resumen muy breve (máximo 50 palabras) explicando por qué.
    Formato de respuesta:
    SENTIMIENTO: [SENTIMIENTO]
    RESUMEN: [Resumen]
    """
    
    try:
        modelo = genai.GenerativeModel("gemini-2.5-flash")
        respuesta = modelo.generate_content(prompt)
        return respuesta.text.strip()
    except:
        return None


def calcular_indicadores(df):
    """Calcula indicadores técnicos básicos."""
    if df.empty:
        return df
    
    # SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # EMA
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    
    return df

def calcular_metricas_periodo(ticker, indice_ticker, periodo_dias, tasa_libre_riesgo):
    """Calcula métricas de rendimiento y riesgo para un periodo específico."""
    try:
        # Descargar datos
        datos_ticker = descargar_datos_historicos(ticker, period="5y", interval="1d")
        datos_indice = descargar_datos_historicos(indice_ticker, period="5y", interval="1d")

        precios_ticker = extraer_precios_columna(datos_ticker).dropna()
        precios_indice = extraer_precios_columna(datos_indice).dropna()

        # Alinear datos
        precios_ticker, precios_indice = precios_ticker.align(precios_indice, join="inner")
        
        # Filtrar por periodo
        if periodo_dias == "YTD":
            inicio = datetime(datetime.now().year, 1, 1)
            pp = precios_ticker[precios_ticker.index >= inicio]
            pi = precios_indice[precios_indice.index >= inicio]
        else:
            pp = precios_ticker.tail(periodo_dias)
            pi = precios_indice.tail(periodo_dias)

        if len(pp) < 10:
            return None

        # Calcular retornos
        rp = pp.pct_change().dropna()
        ri = pi.pct_change().dropna()

        # Métricas
        rendimiento = ((pp.iloc[-1] / pp.iloc[0]) - 1) * 100
        volatilidad = rp.std() * np.sqrt(252) * 100
        beta = np.cov(rp, ri)[0, 1] / np.var(ri) if np.var(ri) != 0 else 0
        rend_ind = ((pi.iloc[-1] / pi.iloc[0]) - 1) * 100
        alpha = rendimiento - (beta * rend_ind)
        
        # Sharpe Ratio con tasa libre de riesgo
        rendimiento_anual = rp.mean() * 252
        sharpe = (rendimiento_anual - tasa_libre_riesgo) / (rp.std() * np.sqrt(252)) if rp.std() != 0 else 0

        return {
            "rendimiento": rendimiento,
            "volatilidad": volatilidad,
            "beta": beta,
            "alpha": alpha,
            "sharpe": sharpe
        }
    except:
        return None


def generar_analisis_ai(prompt):
    """Genera análisis con Gemini si está disponible, sino retorna None."""
    if not GEMINI_DISPONIBLE:
        return None
    
    try:
        modelo = genai.GenerativeModel("gemini-2.5-flash")
        respuesta = modelo.generate_content(prompt)
        return respuesta.text.strip()
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
            return None
        return None


# -----------------------------
# ENCABEZADO PRINCIPAL
# -----------------------------
st.title("📊 Análisis Integral de Acciones")
st.caption("Análisis profesional con Yahoo Finance, Finviz y Gemini AI")

# Mostrar advertencia si Gemini no está disponible
if not GEMINI_DISPONIBLE:
    st.warning("⚠️ **Funcionalidad limitada:** El servicio de IA (Gemini) no está disponible actualmente debido a límite de requests. La app funcionará con todas las métricas y gráficos, pero sin análisis de IA ni traducciones.")

st.markdown("---")

# -----------------------------
# 🔍 BÚSQUEDA O ANÁLISIS
# -----------------------------

if st.session_state["ticker"] is None:
    # MODO BÚSQUEDA
    st.subheader("🔎 Buscar Empresa / Ticker")

    busqueda = st.text_input(
        "Escribe el nombre de la empresa o el ticker:",
        placeholder="Ejemplo: Apple, Tesla, Amazon...",
        key="busqueda_input"
    )

    resultados = []

    if len(busqueda) >= 3:
        with st.spinner("Buscando empresas..."):
            resultados = buscar_empresas_detallado(busqueda)
        resultados = resultados[:3]

    if resultados:
        st.write("### Resultados (Top 3):")

        for item in resultados:
            nombre = item["nombre"]
            pais = item["pais"]
            ticker = item["ticker"]

            try:
                sector = yf.Ticker(ticker).info.get("sector", "N/A")
            except:
                sector = "N/A"

            col1, col2 = st.columns([4,1])

            with col1:
                st.markdown(f"""
                    **{nombre}**  
                    📍 País: {pais}  
                    🏭 Sector: {sector}  
                    🎫 Ticker: `{ticker}`
                """)

            with col2:
                if st.button(f"✅ Seleccionar", key=f"sel_{ticker}"):
                    st.session_state["ticker"] = ticker
                    st.rerun()

else:
    # MODO ANÁLISIS
    ticker_final = st.session_state["ticker"]
    
    # Configuración solo muestra idioma si Gemini está disponible
    if GEMINI_DISPONIBLE:
        col_ticker, col_indice, col_idioma, col_reset = st.columns([3, 2, 2, 1])
    else:
        col_ticker, col_indice, col_reset = st.columns([4, 3, 1])
    
    with col_ticker:
        st.info(f"**📊 Ticker:** {ticker_final}")
    
    with col_indice:
        indices_dict = {
            "S&P 500": "^GSPC",
            "NASDAQ 100": "^NDX",
            "Dow Jones": "^DJI",
            "Russell 2000": "^RUT",
            "IPC México": "^MXX"
        }
        indice_select = st.selectbox("📈 Índice:", list(indices_dict.keys()), key="indice_sel", label_visibility="collapsed")

    if GEMINI_DISPONIBLE:
        with col_idioma:
            idioma = st.selectbox(
                "🌐 Idioma:",
                ["Inglés", "Español", "Francés", "Alemán", "Italiano", "Portugués"],
                key="idioma_sel",
                label_visibility="collapsed"
            )
    else:
        idioma = "Inglés"
    
    with col_reset:
        if st.button("🔄", help="Nueva búsqueda"):
            st.session_state["ticker"] = None
            st.session_state["analizar"] = False
            st.rerun()

    st.markdown("---")

    # BOTÓN ANALIZAR
    if not st.session_state.get("analizar", False):
        if st.button("🚀 Analizar", type="primary", use_container_width=True):
            st.session_state["analizar"] = True
            st.rerun()

    # ======================================================
    # ANÁLISIS COMPLETO
    # ======================================================

    if st.session_state.get("analizar", False):
        
        # Obtener tasa libre de riesgo
        tasa_libre_riesgo = obtener_tasa_libre_riesgo()
        
        try:
            ticker_info = yf.Ticker(ticker_final)
            info = ticker_info.info

            if not info:
                raise ValueError("No se pudo obtener información del ticker")

        except Exception as e:
            st.error(f"❌ No se pudo cargar la información del ticker: {e}")
            st.stop()


        # ==============================
        # FINANCIAL INSIGHTS DE YAHOO FINANCE
        # ==============================
        financial_insights = obtener_financial_insights_yf(ticker_final)

        if financial_insights:
            st.subheader("✨ Insights Financieros")
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                    <p style='margin: 0; line-height: 1.6;'>{financial_insights}</p>
                    <p style='margin-top: 10px; font-size: 11px; opacity: 0.8;'>
                        Powered by Yahoo Finance
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")


        # ==============================
        # INFORMACIÓN GENERAL - DISEÑO MEJORADO
        # ==============================
        st.subheader("🏢 Información General")
        
        # Card principal con información de la empresa
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                        padding: 30px; border-radius: 15px; color: white;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.3); margin-bottom: 25px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h2 style='margin: 0; font-size: 32px; font-weight: bold;'>{info.get('longName', 'N/A')}</h2>
                        <p style='margin: 5px 0; font-size: 16px; opacity: 0.9;'>
                            🏭 {info.get('sector', 'N/A')} • {info.get('industry', 'N/A')}
                        </p>
                        <p style='margin: 5px 0; font-size: 14px; opacity: 0.8;'>
                            📍 {info.get('country', 'N/A')} • 👥 {f"{info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else "N/A"} empleados
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        desc = info.get('longBusinessSummary', 'Descripción no disponible.')
        
        with st.expander("📄 Ver Descripción Completa"):
            if GEMINI_DISPONIBLE:
                with st.spinner(f"Traduciendo a {idioma}..."):
                    desc_trad = traducir_descripcion(desc, idioma)
                st.write(desc_trad)
            else:
                st.write(desc)
                st.caption("ℹ️ Descripción en idioma original (traducción no disponible sin IA)")

        st.markdown("---")


        # ==============================
        # MÉTRICAS BURSÁTILES
        # ==============================
        st.subheader("📈 Métricas Bursátiles")
        
        st.markdown("""
            <style>
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: white;
                box-shadow: 0 6px 20px rgba(0,0,0,0.25);
                margin-bottom: 15px;
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            }
            .metric-label {
                font-size: 13px;
                opacity: 0.9;
                margin-bottom: 8px;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            .metric-value-big {
                font-size: 28px;
                font-weight: bold;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Precio Actual
        precio_actual = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        col1.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💵 Precio Actual</div>
                <div class="metric-value-big">${precio_actual:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Market Cap
        market_cap = info.get('marketCap', 0)
        col2.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🏦 Market Cap</div>
                <div class="metric-value-big">${market_cap/1e9:.1f}B</div>
            </div>
        """, unsafe_allow_html=True)
        
        # P/E Ratio
        pe = info.get('trailingPE', 0)
        col3.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📊 P/E Ratio</div>
                <div class="metric-value-big">{pe:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Beta
        beta = info.get('beta', 0)
        col4.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📉 Beta</div>
                <div class="metric-value-big">{beta:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # EPS
        eps = info.get('trailingEps', 0)
        col5.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💰 EPS</div>
                <div class="metric-value-big">${eps:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

        # Segunda fila
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # 52 Week High
        high_52 = info.get('fiftyTwoWeekHigh', 0)
        col1.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📈 52W High</div>
                <div class="metric-value-big">${high_52:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 52 Week Low
        low_52 = info.get('fiftyTwoWeekLow', 0)
        col2.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📉 52W Low</div>
                <div class="metric-value-big">${low_52:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Volume
        volume = info.get('volume', 0)
        col3.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📊 Volumen</div>
                <div class="metric-value-big">{volume/1e6:.1f}M</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Avg Volume
        avg_vol = info.get('averageVolume', 0)
        col4.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📊 Vol. Promedio</div>
                <div class="metric-value-big">{avg_vol/1e6:.1f}M</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Dividend Yield
        div_yield = info.get('dividendYield', 0)
        col5.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💵 Div. Yield</div>
                <div class="metric-value-big">{div_yield*100:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")


        # ==============================
        # MÉTRICAS CORPORATIVAS
        # ==============================
        st.subheader("🏢 Métricas Corporativas")
        
        st.markdown("""
            <style>
            .corporate-card {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: white;
                box-shadow: 0 6px 20px rgba(0,0,0,0.25);
                margin-bottom: 15px;
                transition: all 0.3s ease;
            }
            .corporate-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            }
            </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Revenue
        revenue = info.get('totalRevenue', 0)
        col1.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">💼 Ventas 12M TTM</div>
                <div class="metric-value-big">${revenue/1e9:.1f}B</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Net Income
        net_income = info.get('netIncomeToCommon', 0)
        col2.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">💵 Utilidad Neta</div>
                <div class="metric-value-big">${net_income/1e9:.1f}B</div>
            </div>
        """, unsafe_allow_html=True)
        
        # ROE
        roe = info.get('returnOnEquity', 0)
        col3.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">📊 ROE</div>
                <div class="metric-value-big">{roe*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Profit Margin
        profit_margin = info.get('profitMargins', 0)
        col4.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">📈 Margen de Ganancia</div>
                <div class="metric-value-big">{profit_margin*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Gross Margin
        gross_margin = info.get('grossMargins', 0)
        col5.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">📊 Margen Bruto</div>
                <div class="metric-value-big">{gross_margin*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

        # Segunda fila
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Operating Margin
        op_margin = info.get('operatingMargins', 0)
        col1.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">📊 Margen de Operación</div>
                <div class="metric-value-big">{op_margin*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # ROA
        roa = info.get('returnOnAssets', 0)
        col2.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">💼 ROA</div>
                <div class="metric-value-big">{roa*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Debt to Equity
        debt_equity = info.get('debtToEquity', 0)
        col3.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">⚖️ Deuda/Equity</div>
                <div class="metric-value-big">{debt_equity/100:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Current Ratio
        current_ratio = info.get('currentRatio', 0)
        col4.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">💧 Radio de Liquidez</div>
                <div class="metric-value-big">{current_ratio:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Free Cash Flow
        fcf = info.get('freeCashflow', 0)
        col5.markdown(f"""
            <div class="corporate-card">
                <div class="metric-label">💰 Free Cash Flow</div>
                <div class="metric-value-big">${fcf/1e9:.1f}B</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")


        # ==============================
        # COMPARACIÓN CON PEERS
        # ==============================
        st.subheader("🔍 Comparación con Competidores")
        peers = obtener_peers_finviz(ticker_final)

        if peers:
            df_comp = obtener_kpis_peers(ticker_final, peers)

            if not df_comp.empty:
                def highlight(row):
                    return ["background-color: #1E8BC3; color: white; font-weight: bold"] * len(row) if row["Ticker"] == ticker_final else [""] * len(row)

                st.dataframe(df_comp.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)

                st.markdown("---")


        # ==============================
        # GRÁFICO DE VELAS
        # ==============================
        st.subheader("📊 Gráfico de Velas")

        # Selector de periodo para el gráfico de velas
        col_periodo, col_espacio = st.columns([2, 4])

        with col_periodo:
            periodo_velas_opciones = {
                "1 Mes": "1mo",
                "3 Meses": "3mo",
                "6 Meses": "6mo",
                "1 Año": "1y",
                "2 Años": "2y",
                "3 Años": "3y",
                "5 Años": "5y"
            }
            
            periodo_velas_sel = st.selectbox(
                "Selecciona el periodo:",
                list(periodo_velas_opciones.keys()),
                index=3,  # Por defecto "1 Año"
                key="periodo_velas"
            )

        datos = descargar_datos_historicos(ticker_final, period=periodo_velas_opciones[periodo_velas_sel], interval="1d")

        if not datos.empty:
            if isinstance(datos.columns, pd.MultiIndex):
                open_col = datos["Open"].iloc[:, 0]
                high_col = datos["High"].iloc[:, 0]
                low_col = datos["Low"].iloc[:, 0]
                close_col = datos["Close"].iloc[:, 0]
                datos_flat = datos.iloc[:, :4].copy() # Simplificación para indicadores
                datos_flat.columns = ["Close", "High", "Low", "Open"] # Ajuste temporal, mejor usar nombres correctos
                # Reconstruir DataFrame plano para indicadores
                df_ind = pd.DataFrame({
                    "Open": open_col,
                    "High": high_col,
                    "Low": low_col,
                    "Close": close_col
                })
            else:
                open_col = datos["Open"]
                high_col = datos["High"]
                low_col = datos["Low"]
                close_col = datos["Close"]
                df_ind = datos.copy()

            # Calcular indicadores
            df_ind = calcular_indicadores(df_ind)

            # Selectores de indicadores
            st.markdown("##### 🛠️ Indicadores Técnicos")
            col_ind1, col_ind2, col_ind3, col_ind4 = st.columns(4)
            with col_ind1:
                show_sma = st.multiselect("SMA", ["20", "50", "200"], key="sma_sel")
            with col_ind2:
                show_ema = st.multiselect("EMA", ["20", "50"], key="ema_sel")
            with col_ind3:
                show_bb = st.checkbox("Bandas Bollinger", key="bb_sel")
            with col_ind4:
                show_osc = st.multiselect("Osciladores", ["RSI", "MACD"], key="osc_sel")

            # Crear figura con subplots si es necesario
            rows = 1
            row_heights = [0.7]
            specs = [[{"secondary_y": False}]]
            
            if "RSI" in show_osc:
                rows += 1
                row_heights.append(0.15)
                specs.append([{"secondary_y": False}])
            if "MACD" in show_osc:
                rows += 1
                row_heights.append(0.15)
                specs.append([{"secondary_y": False}])
            
            # Normalizar alturas
            total_h = sum(row_heights)
            row_heights = [h/total_h for h in row_heights]

            from plotly.subplots import make_subplots
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=row_heights)

            # Velas
            fig.add_trace(go.Candlestick(
                x=df_ind.index,
                open=df_ind["Open"],
                high=df_ind["High"],
                low=df_ind["Low"],
                close=df_ind["Close"],
                name="Precio",
                increasing_line_color="#26A65B",
                decreasing_line_color="#C0392B"
            ), row=1, col=1)

            # SMA
            colors_sma = {"20": "#F1C40F", "50": "#E67E22", "200": "#3498DB"}
            for per in show_sma:
                if f'SMA_{per}' in df_ind.columns:
                    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind[f'SMA_{per}'], mode='lines', 
                                            name=f'SMA {per}', line=dict(color=colors_sma.get(per, "white"), width=1)), row=1, col=1)

            # EMA
            colors_ema = {"20": "#9B59B6", "50": "#8E44AD"}
            for per in show_ema:
                if f'EMA_{per}' in df_ind.columns:
                    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind[f'EMA_{per}'], mode='lines', 
                                            name=f'EMA {per}', line=dict(color=colors_ema.get(per, "white"), width=1, dash='dot')), row=1, col=1)

            # Bollinger Bands
            if show_bb:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['BB_Upper'], mode='lines', name='BB Upper',
                                        line=dict(color='rgba(255, 255, 255, 0.3)', width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['BB_Lower'], mode='lines', name='BB Lower',
                                        line=dict(color='rgba(255, 255, 255, 0.3)', width=1), fill='tonexty', 
                                        fillcolor='rgba(255, 255, 255, 0.05)', showlegend=False), row=1, col=1)

            current_row = 2
            
            # RSI
            if "RSI" in show_osc:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['RSI'], mode='lines', name='RSI', line=dict(color='#E74C3C')), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="gray", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="gray", row=current_row, col=1)
                fig.update_yaxes(title_text="RSI", row=current_row, col=1)
                current_row += 1

            # MACD
            if "MACD" in show_osc:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MACD'], mode='lines', name='MACD', line=dict(color='#3498DB')), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Signal_Line'], mode='lines', name='Signal', line=dict(color='#E67E22')), row=current_row, col=1)
                fig.add_bar(x=df_ind.index, y=df_ind['MACD']-df_ind['Signal_Line'], name='Hist', marker_color='gray', row=current_row, col=1)
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)

            fig.update_layout(template="plotly_dark", height=600 + (150 * (rows-1)), xaxis_rangeslider_visible=False,
                            title=f"Análisis Técnico - {ticker_final} ({periodo_velas_sel})")
            st.plotly_chart(fig, use_container_width=True)
            
            # Botón de descarga para datos históricos
            csv_hist = df_ind.to_csv().encode('utf-8')
            st.download_button("⬇️ Descargar Datos Históricos", csv_hist, f"historico_{ticker_final}.csv", "text/csv", key='dl_hist')

        st.markdown("---")

        # ==============================
        # COMPARACIÓN CONTRA ÍNDICE (BASE 0)
        # ==============================
        st.subheader("📈 Rendimiento Comparativo vs Índice")

        try:
            datos_ticker = descargar_datos_historicos(ticker_final, period="1y", interval="1d")
            indice_t = indices_dict[indice_select]
            datos_indice = descargar_datos_historicos(indice_t, period="1y", interval="1d")

            precios_ticker = extraer_precios_columna(datos_ticker)
            precios_indice = extraer_precios_columna(datos_indice)

            precios_ticker, precios_indice = precios_ticker.align(precios_indice, join="inner")

            # Rendimientos en base 0
            rendimiento_ticker = ((precios_ticker / precios_ticker.iloc[0]) - 1) * 100
            rendimiento_indice = ((precios_indice / precios_indice.iloc[0]) - 1) * 100

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=rendimiento_ticker.index, y=rendimiento_ticker.values,
                mode="lines", name=ticker_final, line=dict(color="#1E8BC3", width=3),
                fill='tonexty', fillcolor='rgba(30, 139, 195, 0.1)'
            ))
            fig_comp.add_trace(go.Scatter(
                x=rendimiento_indice.index, y=rendimiento_indice.values,
                mode="lines", name=indice_select, line=dict(color="#E67E22", width=3, dash="dot")
            ))

            # Línea en y=0
            fig_comp.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            fig_comp.update_layout(
                title=f"{ticker_final} vs {indice_select} (Base 0)", 
                template="plotly_white", 
                height=500,
                yaxis_title="Rendimiento (%)",
                xaxis_title="Fecha",
                hovermode='x unified'
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        except Exception as e:
            st.warning(f"No fue posible generar la comparativa: {e}")

        st.markdown("---")


        # ==============================
        # COMPARACIÓN CON PEERS (GRÁFICO BASE 0)
        # ==============================
        if peers:
            st.subheader("📊 Rendimiento vs Competidores (Último Año)")

            try:
                fig_peers = go.Figure()
                
                # Agregar el ticker principal
                datos_main = descargar_datos_historicos(ticker_final, period="1y", interval="1d")
                precios_main = extraer_precios_columna(datos_main)
                rendimiento_main = ((precios_main / precios_main.iloc[0]) - 1) * 100
                
                fig_peers.add_trace(go.Scatter(
                    x=rendimiento_main.index,
                    y=rendimiento_main.values,
                    mode="lines",
                    name=ticker_final,
                    line=dict(color="#1E8BC3", width=4)
                ))

                # Agregar peers (máximo 5)
                colores_peers = ["#E67E22", "#26A65B", "#8E44AD", "#C0392B", "#F39C12"]
                for i, peer in enumerate(peers[:5]):
                    try:
                        datos_peer = descargar_datos_historicos(peer, period="1y", interval="1d")
                        precios_peer = extraer_precios_columna(datos_peer)
                        
                        if not precios_peer.empty:
                            rendimiento_peer = ((precios_peer / precios_peer.iloc[0]) - 1) * 100
                            
                            fig_peers.add_trace(go.Scatter(
                                x=rendimiento_peer.index,
                                y=rendimiento_peer.values,
                                mode="lines",
                                name=peer,
                                line=dict(color=colores_peers[i % len(colores_peers)], width=2, dash="dot")
                            ))
                    except:
                        continue

                # Línea en y=0
                fig_peers.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                fig_peers.update_layout(
                    title=f"Rendimiento Comparativo: {ticker_final} vs Peers (Base 0)",
                    template="plotly_white",
                    height=550,
                    yaxis_title="Rendimiento (%)",
                    xaxis_title="Fecha",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_peers, use_container_width=True)

            except Exception as e:
                st.warning(f"No fue posible generar la comparativa con peers: {e}")

            st.markdown("---")


        # ==============================
        # MÉTRICAS DE RENDIMIENTO Y RIESGO - VISUAL
        # ==============================
        st.subheader("📊 Métricas de Rendimiento y Riesgo")
        
        # Selector de periodo
        periodo_opciones = {
            "1 Mes": 21,
            "3 Meses": 63,
            "6 Meses": 126,
            "YTD": "YTD",
            "1 Año": 252,
            "3 Años": 756,
            "5 Años": 1260
        }
        
        periodo_sel = st.selectbox(
            "Selecciona el periodo:",
            list(periodo_opciones.keys()),
            index=4,
            key="periodo_metricas"
        )
        
        periodo_dias = periodo_opciones[periodo_sel]
        indice_t = indices_dict[indice_select]
        
        # Calcular métricas
        metricas = calcular_metricas_periodo(ticker_final, indice_t, periodo_dias, tasa_libre_riesgo)
        
        if metricas:
            # CSS para bubbles
            st.markdown("""
                <style>
                .metric-bubble {
                    background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
                    margin-bottom: 10px;
                    transition: transform 0.3s ease;
                }
                .metric-bubble:hover {
                    transform: translateY(-5px);
                }
                .metric-title {
                    font-size: 14px;
                    opacity: 0.85;
                    margin-bottom: 8px;
                    font-weight: 500;
                    letter-spacing: 0.5px;
                }
                .metric-value {
                    font-size: 32px;
                    font-weight: bold;
                    color: white;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Mostrar métricas en cards visuales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.markdown(f"""
                <div class="metric-bubble">
                    <div class="metric-title">Rendimiento</div>
                    <div class="metric-value">{metricas['rendimiento']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            col2.markdown(f"""
                <div class="metric-bubble">
                    <div class="metric-title">Volatilidad</div>
                    <div class="metric-value">{metricas['volatilidad']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            col3.markdown(f"""
                <div class="metric-bubble">
                    <div class="metric-title">Beta</div>
                    <div class="metric-value">{metricas['beta']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            col4.markdown(f"""
                <div class="metric-bubble">
                    <div class="metric-title">Alpha</div>
                    <div class="metric-value">{metricas['alpha']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            col5.markdown(f"""
                <div class="metric-bubble">
                    <div class="metric-title">Sharpe Ratio</div>
                    <div class="metric-value">{metricas['sharpe']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"📊 Métricas calculadas para el periodo: **{periodo_sel}** vs **{indice_select}** | 📈 Tasa libre de riesgo (CETES 28): **{tasa_libre_riesgo*100:.2f}%**")
        else:
            st.warning("No se pudieron calcular las métricas para este periodo.")

        st.markdown("---")


        # ==============================
        # ESTADOS FINANCIEROS
        # ==============================
        st.subheader("� Estados Financieros")
        
        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
        
        with tab1:
            st.caption("Estado de Resultados")
            df_income = obtener_income_yahoo(ticker_final)
            if not df_income.empty:
                st.dataframe(df_income, use_container_width=True, hide_index=True)
                csv = df_income.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar CSV", csv, "income_statement.csv", "text/csv", key='dl_income')
            else:
                st.info("No hay datos disponibles.")
                
        with tab2:
            st.caption("Balance General")
            df_balance = obtener_balance_yahoo(ticker_final)
            if not df_balance.empty:
                st.dataframe(df_balance, use_container_width=True, hide_index=True)
                csv = df_balance.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar CSV", csv, "balance_sheet.csv", "text/csv", key='dl_balance')
            else:
                st.info("No hay datos disponibles.")

        with tab3:
            st.caption("Flujo de Efectivo")
            df_cash = obtener_cashflow_yahoo(ticker_final)
            if not df_cash.empty:
                st.dataframe(df_cash, use_container_width=True, hide_index=True)
                csv = df_cash.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar CSV", csv, "cash_flow.csv", "text/csv", key='dl_cash')
            else:
                st.info("No hay datos disponibles.")

        st.markdown("---")


        # ==============================
        # NOTICIAS Y SENTIMIENTO
        # ==============================
        st.subheader("📰 Noticias y Sentimiento de Mercado")
        
        noticias = obtener_noticias_yf(ticker_final)
        
        if noticias:
            col_news, col_sentiment = st.columns([3, 2])
            
            with col_news:
                st.write("##### Titulares Recientes")
                for n in noticias[:5]:
                    titulo = n.get('title', 'Sin título')
                    link = n.get('link', '#')
                    publisher = n.get('publisher', 'Desconocido')
                    # Manejar diferentes formatos de tiempo si es necesario
                    st.markdown(f"- [{titulo}]({link}) <span style='color:gray; font-size:12px'>({publisher})</span>", unsafe_allow_html=True)
            
            with col_sentiment:
                st.write("##### 🧠 Análisis de Sentimiento (IA)")
                if GEMINI_DISPONIBLE:
                    with st.spinner("Analizando sentimiento..."):
                        sentimiento = analizar_sentimiento_gemini(ticker_final, noticias)
                    
                    if sentimiento:
                        color_bg = "#2C3E50"
                        if "POSITIVO" in sentimiento.upper():
                            color_bg = "linear-gradient(135deg, #26A65B 0%, #2ECC71 100%)"
                        elif "NEGATIVO" in sentimiento.upper():
                            color_bg = "linear-gradient(135deg, #C0392B 0%, #E74C3C 100%)"
                        elif "NEUTRAL" in sentimiento.upper():
                            color_bg = "linear-gradient(135deg, #F39C12 0%, #F1C40F 100%)"
                            
                        st.markdown(f"""
                            <div style='background: {color_bg}; padding: 20px; border-radius: 10px; color: white;'>
                                {sentimiento.replace(chr(10), "<br>")}
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No se pudo generar el análisis de sentimiento.")
                else:
                    st.warning("Análisis de IA no disponible.")
        else:
            st.info("No se encontraron noticias recientes.")

        st.markdown("---")


        # ==============================
        # ANÁLISIS INDIVIDUAL CON GEMINI
        # ==============================
        if GEMINI_DISPONIBLE:
            st.subheader(f"🧠 Análisis Individual con Gemini AI ({idioma})")

            prompt_individual = f"""
Eres un analista financiero profesional. Analiza la empresa {ticker_final} ({info.get('longName', 'N/A')}) con los siguientes datos:

Sector: {info.get('sector')}
Industria: {info.get('industry')}
Market Cap: {info.get('marketCap')}
P/E: {info.get('trailingPE')}
ROE: {info.get('returnOnEquity')}
EPS: {info.get('trailingEps')}
Beta: {info.get('beta')}
Profit Margin: {info.get('profitMargins')}
Gross Margin: {info.get('grossMargins')}
Dividend Yield: {info.get('dividendYield')}

Financial Insight de Yahoo Finance: {financial_insights if financial_insights else "No disponible"}

Genera un análisis profesional en máximo 300 palabras EN {idioma.upper()} con:
1. Recomendación de inversión (Comprar/Vender/No comprar)
2. Fortalezas clave
3. Riesgos principales
4. Valoración actual
5. Perspectiva a corto plazo (3-6 meses)
6. Perspectiva a largo plazo (1-3 años)

Da la respuesta en formato plano, sin asteriscos ni formato markdown. Supón que está pensando en invertir, pero no sabe si comprar, vender, no comprar o no vender, haz diferentes recomendaciones en base de las suposiciones que más recomiendes según las métricas.
"""

            with st.spinner("Generando análisis individual..."):
                analisis_individual = generar_analisis_ai(prompt_individual)

            if analisis_individual:
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 25px; border-radius: 12px; color: white; margin-bottom: 20px;
                                box-shadow: 0 6px 20px rgba(0,0,0,0.3);'>
                        {analisis_individual.replace(chr(10), "<br>")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("🤖 **Análisis de IA no disponible:** El límite de requests de Gemini se ha alcanzado. Por favor, intenta más tarde.")

            st.markdown("---")


            # ==============================
            # ANÁLISIS COMPARATIVO CON PEERS
            # ==============================
            if peers:
                st.subheader(f"🔬 Análisis Comparativo con Competidores ({idioma})")
                
                peers_insights = obtener_financial_insights_peers(peers[:5])
                
                peers_data = []
                for p in peers[:5]:
                    try:
                        p_info = yf.Ticker(p).info
                        peers_data.append({
                            "ticker": p,
                            "name": p_info.get('longName', p),
                            "pe": p_info.get('trailingPE', 'N/A'),
                            "roe": p_info.get('returnOnEquity', 'N/A'),
                            "margin": p_info.get('profitMargins', 'N/A'),
                            "marketcap": p_info.get('marketCap', 'N/A')
                        })
                    except:
                        continue
                
                peers_summary = "\n".join([
                    f"- {p['name']} ({p['ticker']}): P/E={p['pe']}, ROE={p['roe']}, Profit Margin={p['margin']}, Market Cap={p['marketcap']}"
                    for p in peers_data
                ])
                
                prompt_comparativo = f"""
Eres un analista financiero profesional. Compara {ticker_final} ({info.get('longName', 'N/A')}) contra sus principales competidores:

DATOS DE {ticker_final}:
- Market Cap: {info.get('marketCap')}
- P/E: {info.get('trailingPE')}
- ROE: {info.get('returnOnEquity')}
- Profit Margin: {info.get('profitMargins')}
- Gross Margin: {info.get('grossMargins')}

COMPETIDORES:
{peers_summary}

INSIGHTS DE COMPETIDORES:
{chr(10).join([f"- {t}: {insight[:200]}..." for t, insight in peers_insights.items()])}

Genera un análisis comparativo en máximo 300 palabras EN {idioma.upper()} que incluya:
1. Posición competitiva de {ticker_final} en el sector
2. Ventajas competitivas vs peers
3. Desventajas o áreas de mejora
4. Valoración relativa (sobrevalorada/infravalorada vs peers)
5. Recomendación comparativa

Da la respuesta en formato plano, sin asteriscos ni formato markdown. Hasta el final recomienda de los peers y el ticker analizado, en que orden invertirías, enuméralos. Pon en mayúsculas la recomendación de comprar o vender. QUE SEA MUY CLARO. 
"""

                with st.spinner("Generando análisis comparativo con peers..."):
                    analisis_comparativo = generar_analisis_ai(prompt_comparativo)

                if analisis_comparativo:
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                    padding: 25px; border-radius: 12px; color: white;
                                    box-shadow: 0 6px 20px rgba(0,0,0,0.3);'>
                            {analisis_comparativo.replace(chr(10), "<br>")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.info("🤖 **Análisis comparativo de IA no disponible:** El límite de requests de Gemini se ha alcanzado. Por favor, intenta más tarde.")

        else:
            st.info("🤖 **Análisis de IA no disponible:** El servicio de Gemini AI está temporalmente deshabilitado debido al límite de requests. Todos los gráficos, métricas y datos financieros están disponibles y funcionando correctamente.")

        st.markdown("---")
        st.warning("⚠️ Esto no es recomendación financiera. Solo fines educativos.")

        # ==============================
        # FOOTER
        # ==============================
        st.markdown("""
        <div style='text-align:center; color:gray; font-size:11px; margin-top: 40px;'>
        📊 <b>Fuentes:</b> Yahoo Finance, Finviz & Banxico | 🤖 <b>IA:</b> Gemini 2.5 Flash<br>
        🎓 Ingeniería Financiera | 💻 Versión 4.3 | ⚖️ Solo para uso educativo
        </div>
        """, unsafe_allow_html=True)