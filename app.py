import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
import re
import time

# -----------------------------
# CONFIGURACIÓN INICIAL
# -----------------------------
st.set_page_config(page_title="Análisis de Acciones", page_icon="📊", layout="wide")

# Configurar Gemini
GEMINI_DISPONIBLE = False
GEMINI_ERROR_INIT = None

try:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        GEMINI_DISPONIBLE = True
    else:
        GEMINI_ERROR_INIT = "No se encontró GEMINI_API_KEY en secrets"
except Exception as e:
    GEMINI_ERROR_INIT = str(e)

# Session State
if "ticker" not in st.session_state:
    st.session_state["ticker"] = None
if "analizar" not in st.session_state:
    st.session_state["analizar"] = False

# Headers para requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# -----------------------------
# FUNCIONES DE SCRAPING
# -----------------------------

@st.cache_data(ttl=600)
def obtener_tasa_libre_riesgo():
    """Obtiene la tasa de CETES a 28 días desde Banxico."""
    try:
        url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43936/datos/oportuno"
        headers = {"Bmx-Token": "TU_TOKEN_AQUI"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            tasa = float(data['bmx']['series'][0]['datos'][0]['dato'])
            return tasa / 100
    except:
        pass
    return 0.10

@st.cache_data(ttl=600)
def scrape_yahoo_summary(ticker):
    """Scrapea datos generales de Yahoo Finance."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        info = {}
        
        # Precio actual
        try:
            price_elem = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
            if price_elem:
                info['currentPrice'] = float(price_elem.text.replace(',', ''))
        except:
            info['currentPrice'] = 0
        
        # Nombre de la empresa
        try:
            name_elem = soup.find('h1', {'class': 'yf-xxbei9'})
            if name_elem:
                info['longName'] = name_elem.text.split('(')[0].strip()
        except:
            info['longName'] = ticker
        
        # Tabla de datos
        try:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        label = cols[0].text.strip()
                        value = cols[1].text.strip()
                        
                        if label == 'Market Cap':
                            info['marketCap'] = parse_value(value)
                        elif label == 'PE Ratio (TTM)' or label == 'P/E Ratio':
                            info['trailingPE'] = parse_float(value)
                        elif label == 'Beta (5Y Monthly)' or label == 'Beta':
                            info['beta'] = parse_float(value)
                        elif label == 'EPS (TTM)':
                            info['trailingEps'] = parse_float(value)
                        elif label == '52 Week High':
                            info['fiftyTwoWeekHigh'] = parse_float(value)
                        elif label == '52 Week Low':
                            info['fiftyTwoWeekLow'] = parse_float(value)
                        elif label == 'Volume' or label == 'Avg. Volume':
                            if 'volume' not in info:
                                info['volume'] = parse_value(value)
                            else:
                                info['averageVolume'] = parse_value(value)
                        elif label == 'Forward Dividend & Yield' or label == 'Dividend Yield':
                            try:
                                if '%' in value:
                                    info['dividendYield'] = parse_float(value.split('(')[1].replace('%)', '')) / 100
                                else:
                                    info['dividendYield'] = parse_float(value) / 100
                            except:
                                info['dividendYield'] = 0
        except:
            pass
        
        return info
    except Exception as e:
        st.error(f"Error scraping summary: {e}")
        return {}

@st.cache_data(ttl=600)
def scrape_yahoo_statistics(ticker):
    """Scrapea estadísticas detalladas de Yahoo Finance."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        stats = {}
        
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    label = cols[0].text.strip()
                    value = cols[1].text.strip()
                    
                    if 'Return on Equity' in label or 'ROE' in label:
                        stats['returnOnEquity'] = parse_float(value) / 100
                    elif 'Return on Assets' in label or 'ROA' in label:
                        stats['returnOnAssets'] = parse_float(value) / 100
                    elif 'Profit Margin' in label:
                        stats['profitMargins'] = parse_float(value) / 100
                    elif 'Gross Margin' in label:
                        stats['grossMargins'] = parse_float(value) / 100
                    elif 'Operating Margin' in label:
                        stats['operatingMargins'] = parse_float(value) / 100
                    elif 'Total Debt/Equity' in label:
                        stats['debtToEquity'] = parse_float(value)
                    elif 'Current Ratio' in label:
                        stats['currentRatio'] = parse_float(value)
        
        return stats
    except Exception as e:
        st.error(f"Error scraping statistics: {e}")
        return {}

@st.cache_data(ttl=600)
def scrape_yahoo_profile(ticker):
    """Scrapea perfil de la empresa de Yahoo Finance."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/profile"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        profile = {}
        
        # Descripción
        try:
            desc_section = soup.find('section', {'data-testid': 'description'})
            if desc_section:
                desc_p = desc_section.find('p')
                if desc_p:
                    profile['longBusinessSummary'] = desc_p.text.strip()
        except:
            pass
        
        # Datos de la tabla
        try:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        label = cols[0].text.strip()
                        value = cols[1].text.strip()
                        
                        if 'Sector' in label:
                            profile['sector'] = value
                        elif 'Industry' in label:
                            profile['industry'] = value
                        elif 'Full Time Employees' in label or 'Employees' in label:
                            profile['fullTimeEmployees'] = parse_value(value)
                        elif 'Country' in label:
                            profile['country'] = value
        except:
            pass
        
        return profile
    except Exception as e:
        st.error(f"Error scraping profile: {e}")
        return {}

@st.cache_data(ttl=600)
def scrape_yahoo_financials(ticker):
    """Scrapea estados financieros de Yahoo Finance."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/financials"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        financials = {}
        
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    label = cols[0].text.strip()
                    value = cols[1].text.strip()
                    
                    if 'Total Revenue' in label:
                        financials['totalRevenue'] = parse_value(value)
                    elif 'Net Income' in label and 'Common' in label:
                        financials['netIncomeToCommon'] = parse_value(value)
                    elif 'Free Cash Flow' in label or 'Operating Cash Flow' in label:
                        if 'freeCashflow' not in financials:
                            financials['freeCashflow'] = parse_value(value)
        
        return financials
    except Exception as e:
        st.error(f"Error scraping financials: {e}")
        return {}

@st.cache_data(ttl=3600)
def scrape_historical_data(ticker, period_days=365):
    """Scrapea datos históricos de Yahoo Finance."""
    try:
        end_timestamp = int(time.time())
        start_timestamp = end_timestamp - (period_days * 24 * 60 * 60)
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_timestamp}&period2={end_timestamp}&interval=1d&events=history"
        
        response = requests.get(url, headers=HEADERS, timeout=15)
        
        if response.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error scraping historical data: {e}")
        return pd.DataFrame()

def parse_value(value_str):
    """Convierte valores como '1.5T', '234.5B', '45.6M' a números."""
    try:
        value_str = value_str.replace(',', '').replace('$', '').strip()
        
        if 'T' in value_str:
            return float(value_str.replace('T', '')) * 1e12
        elif 'B' in value_str:
            return float(value_str.replace('B', '')) * 1e9
        elif 'M' in value_str:
            return float(value_str.replace('M', '')) * 1e6
        elif 'K' in value_str:
            return float(value_str.replace('K', '')) * 1e3
        else:
            return float(value_str)
    except:
        return 0

def parse_float(value_str):
    """Convierte string a float."""
    try:
        return float(value_str.replace(',', '').replace('%', '').replace('$', '').strip())
    except:
        return 0

def get_complete_info(ticker):
    """Obtiene información completa combinando todos los scrapers."""
    info = {}
    
    # Summary
    summary = scrape_yahoo_summary(ticker)
    info.update(summary)
    
    # Statistics
    stats = scrape_yahoo_statistics(ticker)
    info.update(stats)
    
    # Profile
    profile = scrape_yahoo_profile(ticker)
    info.update(profile)
    
    # Financials
    financials = scrape_yahoo_financials(ticker)
    info.update(financials)
    
    return info

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------

@st.cache_data(ttl=600)
def buscar_empresas_detallado(query):
    """Busca empresas usando Yahoo Finance Search."""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            quotes = data.get("quotes", [])
            
            resultados = []
            for q in quotes:
                if q.get("isYahooFinance", False):
                    resultados.append({
                        "ticker": q.get("symbol", "N/A"),
                        "nombre": q.get("longname") or q.get("shortname", "N/A"),
                        "pais": q.get("exchDisp", "N/A")
                    })
            
            return resultados
    except:
        pass
    return []

@st.cache_data(ttl=600)
def obtener_peers_finviz(ticker):
    """Obtiene competidores desde Finviz."""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            peers_cell = soup.find("td", string="Industry")
            
            if peers_cell:
                peers_row = peers_cell.find_next_sibling("td")
                if peers_row:
                    links = peers_row.find_all("a")
                    peers = [link.text.strip() for link in links if link.text.strip()]
                    return peers[:5]
    except:
        pass
    return []

@st.cache_data(ttl=600)
def obtener_kpis_peers(ticker_actual, peers):
    """Obtiene KPIs de peers para comparación."""
    try:
        tickers = [ticker_actual] + peers[:5]
        data = []
        
        for t in tickers:
            try:
                info = get_complete_info(t)
                data.append({
                    "Ticker": t,
                    "Empresa": info.get("longName", t),
                    "Market Cap (B)": f"${info.get('marketCap', 0)/1e9:.1f}",
                    "P/E": f"{info.get('trailingPE', 0):.2f}",
                    "ROE (%)": f"{info.get('returnOnEquity', 0)*100:.1f}",
                    "Profit Margin (%)": f"{info.get('profitMargins', 0)*100:.1f}",
                    "Beta": f"{info.get('beta', 0):.2f}"
                })
            except:
                continue
        
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def scrape_yahoo_income_statement(ticker):
    """Scrapea Income Statement."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/financials"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extraer tabla
        tables = soup.find_all('table')
        if tables:
            dfs = pd.read_html(str(tables[0]))
            if dfs:
                df = dfs[0]
                return df
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=600)
def scrape_yahoo_balance_sheet(ticker):
    """Scrapea Balance Sheet."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/balance-sheet"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        tables = soup.find_all('table')
        if tables:
            dfs = pd.read_html(str(tables[0]))
            if dfs:
                df = dfs[0]
                return df
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=600)
def scrape_yahoo_cash_flow(ticker):
    """Scrapea Cash Flow."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/cash-flow"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        tables = soup.find_all('table')
        if tables:
            dfs = pd.read_html(str(tables[0]))
            if dfs:
                df = dfs[0]
                return df
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=600)
def scrape_yahoo_news(ticker):
    """Scrapea noticias de Yahoo Finance."""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news = []
        articles = soup.find_all('h3')[:10]
        
        for article in articles:
            link_tag = article.find('a')
            if link_tag:
                title = link_tag.text.strip()
                link = link_tag.get('href', '')
                if link and not link.startswith('http'):
                    link = 'https://finance.yahoo.com' + link
                
                news.append({
                    'title': title,
                    'link': link,
                    'publisher': 'Yahoo Finance'
                })
        
        return news
    except:
        return []

def traducir_descripcion(texto, idioma):
    """Traduce texto usando Gemini."""
    if not GEMINI_DISPONIBLE or not texto:
        return texto
    
    try:
        modelo = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"Traduce el siguiente texto a {idioma}, mantén el mismo tono profesional:\n\n{texto}"
        respuesta = modelo.generate_content(prompt)
        
        if hasattr(respuesta, 'text') and respuesta.text:
            return respuesta.text.strip()
        return texto
    except:
        return texto

def analizar_sentimiento_gemini(ticker, noticias):
    """Analiza sentimiento de noticias con Gemini."""
    if not GEMINI_DISPONIBLE or not noticias:
        return None
    
    try:
        titulares = "\n".join([f"- {n.get('title', '')}" for n in noticias[:5]])
        
        prompt = f"""Analiza el sentimiento de mercado para {ticker} basado en estos titulares:

{titulares}

Proporciona en máximo 100 palabras:
1. Sentimiento general (POSITIVO/NEGATIVO/NEUTRAL)
2. Temas clave identificados
3. Implicación para inversores

Responde en formato plano sin markdown."""

        modelo = genai.GenerativeModel("gemini-2.5-flash")
        respuesta = modelo.generate_content(prompt)
        
        if hasattr(respuesta, 'text') and respuesta.text:
            return respuesta.text.strip()
        return None
    except:
        return None

def calcular_indicadores(df):
    """Calcula indicadores técnicos básicos."""
    if df.empty:
        return df
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    
    return df

def calcular_metricas_periodo(ticker, indice_ticker, periodo_dias, tasa_libre_riesgo):
    """Calcula métricas de rendimiento y riesgo para un periodo específico."""
    try:
        datos_ticker = scrape_historical_data(ticker, 1825)
        datos_indice = scrape_historical_data(indice_ticker, 1825)

        if datos_ticker.empty or datos_indice.empty:
            return None

        precios_ticker = datos_ticker['Close'].dropna()
        precios_indice = datos_indice['Close'].dropna()

        precios_ticker, precios_indice = precios_ticker.align(precios_indice, join="inner")
        
        if periodo_dias == "YTD":
            inicio = datetime(datetime.now().year, 1, 1)
            pp = precios_ticker[precios_ticker.index >= inicio]
            pi = precios_indice[precios_indice.index >= inicio]
        else:
            pp = precios_ticker.tail(periodo_dias)
            pi = precios_indice.tail(periodo_dias)

        if len(pp) < 10:
            return None

        rp = pp.pct_change().dropna()
        ri = pi.pct_change().dropna()

        rendimiento = ((pp.iloc[-1] / pp.iloc[0]) - 1) * 100
        volatilidad = rp.std() * np.sqrt(252) * 100
        beta = np.cov(rp, ri)[0, 1] / np.var(ri) if np.var(ri) != 0 else 0
        rend_ind = ((pi.iloc[-1] / pi.iloc[0]) - 1) * 100
        alpha = rendimiento - (beta * rend_ind)
        
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
    """Genera análisis con AI si está disponible, sino retorna None."""
    if not GEMINI_DISPONIBLE:
        return None
    
    try:
        modelo = genai.GenerativeModel("gemini-2.5-flash")
        respuesta = modelo.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )
        
        if hasattr(respuesta, 'text') and respuesta.text:
            return respuesta.text.strip()
        return None
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
            st.session_state['ai_error'] = "Límite de cuota alcanzado"
        elif "api_key" in error_msg.lower() or "invalid" in error_msg.lower():
            st.session_state['ai_error'] = "API Key inválida"
        else:
            st.session_state['ai_error'] = error_msg
        return None

# -----------------------------
# ENCABEZADO PRINCIPAL
# -----------------------------
st.title("📊 Análisis Integral de Acciones")
st.caption("Análisis profesional con Yahoo Finance, Finviz y Gemini AI")

if not GEMINI_DISPONIBLE:
    if GEMINI_ERROR_INIT:
        st.error(f"⚠️ **IA no disponible:** {GEMINI_ERROR_INIT}")
        with st.expander("ℹ️ Ver información de depuración"):
            st.markdown("""
            **Pasos para configurar la API Key en Streamlit Cloud:**
            
            1. Ve a tu app en Streamlit Cloud
            2. Click en **Settings** (⚙️) en la esquina superior derecha
            3. Click en **Secrets** en el menú lateral
            4. Agrega el siguiente contenido (exactamente con este formato):
```toml
            GEMINI_API_KEY = "tu-api-key-aquí"
```
            
            **IMPORTANTE:**
            - El nombre debe ser GEMINI_API_KEY (sin comillas)
            - Debe haber UN ESPACIO antes y después del signo =
            - El valor de la API key DEBE estar entre comillas
            - No debe haber otros caracteres o espacios adicionales
            
            5. Click en **Save**
            6. La app se reiniciará automáticamente
            """)
    else:
        st.warning("⚠️ **Funcionalidad limitada:** El servicio de IA (Gemini) no está disponible actualmente.")

st.markdown("---")

# -----------------------------
# 🔍 BÚSQUEDA O ANÁLISIS
# -----------------------------

if st.session_state["ticker"] is None:
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
                info = get_complete_info(ticker)
                sector = info.get("sector", "N/A")
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
    ticker_final = st.session_state["ticker"]
    
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

    if not st.session_state.get("analizar", False):
        if st.button("🚀 Analizar", type="primary", use_container_width=True):
            st.session_state["analizar"] = True
            st.rerun()

    if st.session_state.get("analizar", False):
        
        tasa_libre_riesgo = obtener_tasa_libre_riesgo()
        
        try:
            with st.spinner("Obteniendo información de la empresa..."):
                info = get_complete_info(ticker_final)

            if not info:
                raise ValueError("No se pudo obtener información del ticker")

        except Exception as e:
            st.error(f"❌ No se pudo cargar la información del ticker: {e}")
            st.stop()

        financial_insights = info.get('longBusinessSummary', '')[:500]

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

        st.subheader("🏢 Información General")
        
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
                position: relative;
            }
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            }
            .metric-card[data-tooltip]:hover::after {
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background-color: #333;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                white-space: normal;
                width: 200px;
                z-index: 1000;
                opacity: 0;
                animation: fadeIn 0.3s forwards;
                pointer-events: none;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                margin-bottom: 10px;
            }
            @keyframes fadeIn {
                to { opacity: 1; }
            }
            .metric-card[data-tooltip]:hover::before {
                content: '';
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                border-width: 6px;
                border-style: solid;
                border-color: #333 transparent transparent transparent;
                margin-bottom: -2px;
                opacity: 0;
                animation: fadeIn 0.3s forwards;
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
        
        tooltips = {
            "precio": "El precio actual de una acción en el mercado.",
            "market_cap": "Capitalización de Mercado: Valor total de todas las acciones de la empresa (Precio x Acciones en circulación).",
            "pe": "Price-to-Earnings Ratio: Mide cuánto pagan los inversores por cada dólar de ganancia. Un P/E alto puede indicar crecimiento o sobrevaloración.",
            "beta": "Mide la volatilidad de una acción respecto al mercado. Beta > 1 es más volátil, Beta < 1 es menos volátil.",
            "eps": "Earnings Per Share (Beneficio por Acción): Ganancia neta dividida por el número de acciones.",
            "high52": "El precio más alto alcanzado en las últimas 52 semanas.",
            "low52": "El precio más bajo alcanzado en las últimas 52 semanas.",
            "volumen": "Número de acciones negociadas en el día actual.",
            "avg_vol": "Promedio de acciones negociadas diariamente en los últimos 3 meses.",
            "div_yield": "Dividend Yield: Porcentaje del precio de la acción que se paga como dividendos anualmente."
        }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        precio_actual = info.get('currentPrice', 0)
        col1.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['precio']}">
                <div class="metric-label">💵 Precio Actual</div>
                <div class="metric-value-big">${precio_actual:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        market_cap = info.get('marketCap', 0)
        col2.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['market_cap']}">
                <div class="metric-label">🏦 Market Cap</div>
                <div class="metric-value-big">${market_cap/1e9:.1f}B</div>
            </div>
        """, unsafe_allow_html=True)
        
        pe = info.get('trailingPE', 0)
        col3.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['pe']}">
                <div class="metric-label">📊 P/E Ratio</div>
                <div class="metric-value-big">{pe:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        beta = info.get('beta', 0)
        col4.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['beta']}">
                <div class="metric-label">📉 Beta</div>
                <div class="metric-value-big">{beta:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        eps = info.get('trailingEps', 0)
        col5.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['eps']}">
                <div class="metric-label">💰 EPS</div>
                <div class="metric-value-big">${eps:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        high_52 = info.get('fiftyTwoWeekHigh', 0)
        col1.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['high52']}">
                <div class="metric-label">📈 52W High</div>
                <div class="metric-value-big">${high_52:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        low_52 = info.get('fiftyTwoWeekLow', 0)
        col2.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['low52']}">
                <div class="metric-label">📉 52W Low</div>
                <div class="metric-value-big">${low_52:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        volume = info.get('volume', 0)
        col3.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['volumen']}">
                <div class="metric-label">📊 Volumen</div>
                <div class="metric-value-big">{volume/1e6:.1f}M</div>
            </div>
        """, unsafe_allow_html=True)
        
        avg_vol = info.get('averageVolume', 0)
        col4.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['avg_vol']}">
                <div class="metric-label">📊 Vol. Promedio</div>
                <div class="metric-value-big">{avg_vol/1e6:.1f}M</div>
            </div>
        """, unsafe_allow_html=True)
        
        div_yield = info.get('dividendYield', 0)
        col5.markdown(f"""
            <div class="metric-card" data-tooltip="{tooltips['div_yield']}">
                <div class="metric-label">💵 Div. Yield</div>
                <div class="metric-value-big">{div_yield*100:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

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
                position: relative;
            }
            .corporate-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            }
            .corporate-card[data-tooltip]:hover::after {
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background-color: #333;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                white-space: normal;
                width: 200px;
                z-index: 1000;
                opacity: 0;
                animation: fadeIn 0.3s forwards;
                pointer-events: none;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                margin-bottom: 10px;
            }
            .corporate-card[data-tooltip]:hover::before {
                content: '';
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                border-width: 6px;
                border-style: solid;
                border-color: #333 transparent transparent transparent;
                margin-bottom: -2px;
                opacity: 0;
                animation: fadeIn 0.3s forwards;
            }
            </style>
        """, unsafe_allow_html=True)
        
        corp_tooltips = {
            "revenue": "Ingresos totales de la empresa en los últimos 12 meses (TTM).",
            "net_income": "Utilidad Neta: Ganancia total después de todos los gastos e impuestos.",
            "roe": "Return on Equity: Mide la rentabilidad generada con el dinero de los accionistas.",
            "profit_margin": "Margen de Ganancia: Porcentaje de ingresos que se convierte en ganancia neta.",
            "gross_margin": "Margen Bruto: Porcentaje de ingresos que queda después de restar el costo de ventas.",
            "op_margin": "Margen Operativo: Porcentaje de ingresos después de gastos operativos pero antes de impuestos.",
            "roa": "Return on Assets: Mide qué tan eficiente es la empresa usando sus activos para generar ganancias.",
            "debt_equity": "Relación Deuda/Capital: Mide el apalancamiento financiero. Valores altos indican mayor riesgo.",
            "current_ratio": "Radio de Liquidez: Capacidad de la empresa para pagar sus deudas a corto plazo (Activo Cte / Pasivo Cte).",
            "fcf": "Free Cash Flow: Efectivo generado por la empresa después de mantener sus activos (CAPEX)."
        }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        revenue = info.get('totalRevenue', 0)
        col1.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['revenue']}">
                <div class="metric-label">💼 Ventas 12M TTM</div>
                <div class="metric-value-big">${revenue/1e9:.1f}B</div>
            </div>
        """, unsafe_allow_html=True)
        
        net_income = info.get('netIncomeToCommon', 0)
        col2.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['net_income']}">
                <div class="metric-label">💵 Utilidad Neta</div>
                <div class="metric-value-big">${net_income/1e9:.1f}B</div>
            </div>
        """, unsafe_allow_html=True)
        
        roe = info.get('returnOnEquity', 0)
        col3.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['roe']}">
                <div class="metric-label">📊 ROE</div>
                <div class="metric-value-big">{roe*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        profit_margin = info.get('profitMargins', 0)
        col4.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['profit_margin']}">
                <div class="metric-label">📈 Margen de Ganancia</div>
                <div class="metric-value-big">{profit_margin*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        gross_margin = info.get('grossMargins', 0)
        col5.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['gross_margin']}">
                <div class="metric-label">📊 Margen Bruto</div>
                <div class="metric-value-big">{gross_margin*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        
        op_margin = info.get('operatingMargins', 0)
        col1.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['op_margin']}">
                <div class="metric-label">📊 Margen de Operación</div>
                <div class="metric-value-big">{op_margin*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        roa = info.get('returnOnAssets', 0)
        col2.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['roa']}">
                <div class="metric-label">💼 ROA</div>
                <div class="metric-value-big">{roa*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        debt_equity = info.get('debtToEquity', 0)
        col3.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['debt_equity']}">
                <div class="metric-label">⚖️ Deuda/Equity</div>
                <div class="metric-value-big">{debt_equity/100:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        current_ratio = info.get('currentRatio', 0)
        col4.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['current_ratio']}">
                <div class="metric-label">💧 Radio de Liquidez</div>
                <div class="metric-value-big">{current_ratio:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        fcf = info.get('freeCashflow', 0)
        col5.markdown(f"""
            <div class="corporate-card" data-tooltip="{corp_tooltips['fcf']}">
                <div class="metric-label">💰 Free Cash Flow</div>
                <div class="metric-value-big">${fcf/1e9:.1f}B</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("🔍 Comparación con Competidores")
        peers = obtener_peers_finviz(ticker_final)

        if peers:
            with st.spinner("Obteniendo datos de competidores..."):
                df_comp = obtener_kpis_peers(ticker_final, peers)

            if not df_comp.empty:
                def highlight(row):
                    return ["background-color: #1E8BC3; color: white; font-weight: bold"] * len(row) if row["Ticker"] == ticker_final else [""] * len(row)

                st.dataframe(df_comp.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)

                st.markdown("---")

        st.subheader("📊 Gráfico de Velas")

        col_periodo, col_espacio = st.columns([2, 4])

        with col_periodo:
            periodo_velas_opciones = {
                "1 Mes": 30,
                "3 Meses": 90,
                "6 Meses": 180,
                "1 Año": 365,
                "2 Años": 730,
                "3 Años": 1095,
                "5 Años": 1825
            }
            
            periodo_velas_sel = st.selectbox(
                "Selecciona el periodo:",
                list(periodo_velas_opciones.keys()),
                index=3,
                key="periodo_velas"
            )

        with st.spinner("Descargando datos históricos..."):
            datos = scrape_historical_data(ticker_final, periodo_velas_opciones[periodo_velas_sel])

        if not datos.empty:
            df_ind = datos.copy()
            df_ind = calcular_indicadores(df_ind)

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
            
            total_h = sum(row_heights)
            row_heights = [h/total_h for h in row_heights]

            from plotly.subplots import make_subplots
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=row_heights)

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

            colors_sma = {"20": "#F1C40F", "50": "#E67E22", "200": "#3498DB"}
            for per in show_sma:
                if f'SMA_{per}' in df_ind.columns:
                    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind[f'SMA_{per}'], mode='lines', 
                                            name=f'SMA {per}', line=dict(color=colors_sma.get(per, "white"), width=1)), row=1, col=1)

            colors_ema = {"20": "#9B59B6", "50": "#8E44AD"}
            for per in show_ema:
                if f'EMA_{per}' in df_ind.columns:
                    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind[f'EMA_{per}'], mode='lines', 
                                            name=f'EMA {per}', line=dict(color=colors_ema.get(per, "white"), width=1, dash='dot')), row=1, col=1)

            if show_bb:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['BB_Upper'], mode='lines', name='BB Upper',
                                        line=dict(color='rgba(255, 255, 255, 0.3)', width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['BB_Lower'], mode='lines', name='BB Lower',
                                        line=dict(color='rgba(255, 255, 255, 0.3)', width=1), fill='tonexty', 
                                        fillcolor='rgba(255, 255, 255, 0.05)', showlegend=False), row=1, col=1)

            current_row = 2
            
            if "RSI" in show_osc:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['RSI'], mode='lines', name='RSI', line=dict(color='#E74C3C')), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="gray", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="gray", row=current_row, col=1)
                fig.update_yaxes(title_text="RSI", row=current_row, col=1)
                current_row += 1

            if "MACD" in show_osc:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MACD'], mode='lines', name='MACD', line=dict(color='#3498DB')), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Signal_Line'], mode='lines', name='Signal', line=dict(color='#E67E22')), row=current_row, col=1)
                fig.add_bar(x=df_ind.index, y=df_ind['MACD']-df_ind['Signal_Line'], name='Hist', marker_color='gray', row=current_row, col=1)
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)

            fig.update_layout(template="plotly_dark", height=600 + (150 * (rows-1)), xaxis_rangeslider_visible=False,
                            title=f"Análisis Técnico - {ticker_final} ({periodo_velas_sel})")
            st.plotly_chart(fig, use_container_width=True)
            
            csv_hist = df_ind.to_csv().encode('utf-8')
            st.download_button("⬇️ Descargar Datos Históricos", csv_hist, f"historico_{ticker_final}.csv", "text/csv", key='dl_hist')

        st.markdown("---")

        st.subheader("📈 Rendimiento Comparativo vs Índice")

        try:
            with st.spinner("Calculando rendimientos..."):
                datos_ticker = scrape_historical_data(ticker_final, 365)
                indice_t = indices_dict[indice_select]
                datos_indice = scrape_historical_data(indice_t, 365)

            if not datos_ticker.empty and not datos_indice.empty:
                precios_ticker = datos_ticker['Close']
                precios_indice = datos_indice['Close']

                precios_ticker, precios_indice = precios_ticker.align(precios_indice, join="inner")

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

        if peers:
            st.subheader("📊 Rendimiento vs Competidores (Último Año)")

            try:
                with st.spinner("Comparando con competidores..."):
                    fig_peers = go.Figure()
                    
                    datos_main = scrape_historical_data(ticker_final, 365)
                    precios_main = datos_main['Close']
                    rendimiento_main = ((precios_main / precios_main.iloc[0]) - 1) * 100
                    
                    fig_peers.add_trace(go.Scatter(
                        x=rendimiento_main.index,
                        y=rendimiento_main.values,
                        mode="lines",
                        name=ticker_final,
                        line=dict(color="#1E8BC3", width=4)
                    ))

                    colores_peers = ["#E67E22", "#26A65B", "#8E44AD", "#C0392B", "#F39C12"]
                    for i, peer in enumerate(peers[:5]):
                        try:
                            datos_peer = scrape_historical_data(peer, 365)
                            precios_peer = datos_peer['Close']
                            
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

        st.subheader("📊 Métricas de Rendimiento y Riesgo")
        
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
        
        with st.spinner("Calculando métricas..."):
            metricas = calcular_metricas_periodo(ticker_final, indice_t, periodo_dias, tasa_libre_riesgo)
        
        if metricas:
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

        st.subheader("📋 Estados Financieros")
        
        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
        
        with tab1:
            st.caption("Estado de Resultados")
            with st.spinner("Obteniendo Income Statement..."):
                df_income = scrape_yahoo_income_statement(ticker_final)
            if not df_income.empty:
                st.dataframe(df_income, use_container_width=True, hide_index=True)
                csv = df_income.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar CSV", csv, "income_statement.csv", "text/csv", key='dl_income')
            else:
                st.info("No hay datos disponibles.")
                
        with tab2:
            st.caption("Balance General")
            with st.spinner("Obteniendo Balance Sheet..."):
                df_balance = scrape_yahoo_balance_sheet(ticker_final)
            if not df_balance.empty:
                st.dataframe(df_balance, use_container_width=True, hide_index=True)
                csv = df_balance.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar CSV", csv, "balance_sheet.csv", "text/csv", key='dl_balance')
            else:
                st.info("No hay datos disponibles.")

        with tab3:
            st.caption("Flujo de Efectivo")
            with st.spinner("Obteniendo Cash Flow..."):
                df_cash = scrape_yahoo_cash_flow(ticker_final)
            if not df_cash.empty:
                st.dataframe(df_cash, use_container_width=True, hide_index=True)
                csv = df_cash.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar CSV", csv, "cash_flow.csv", "text/csv", key='dl_cash')
            else:
                st.info("No hay datos disponibles.")

        st.markdown("---")

        st.subheader("📰 Noticias y Sentimiento de Mercado")
        
        with st.spinner("Obteniendo noticias..."):
            noticias = scrape_yahoo_news(ticker_final)
        
        if noticias:
            col_news, col_sentiment = st.columns([3, 2])
            
            with col_news:
                st.write("##### Titulares Recientes")
                for n in noticias[:5]:
                    titulo = n.get('title', 'Sin título')
                    link = n.get('link', '#')
                    publisher = n.get('publisher', 'Desconocido')
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
                        error_msg = st.session_state.get('ai_error', '')
                        
                        if 'cuota' in error_msg.lower() or 'quota' in error_msg.lower():
                            st.info("🤖 Límite de cuota alcanzado. Intenta más tarde.")
                        elif 'api_key' in error_msg.lower():
                            st.warning("🤖 Error de configuración de API Key.")
                        else:
                            st.info("🤖 No se pudo generar el análisis de sentimiento.")
                else:
                    st.warning("Análisis de IA no disponible.")
        else:
            st.info("No se encontraron noticias recientes.")

        st.markdown("---")

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
                error_msg = st.session_state.get('ai_error', '')
                
                if 'cuota' in error_msg.lower() or 'quota' in error_msg.lower():
                    st.info("🤖 Límite de requests alcanzado. Intenta más tarde.")
                elif 'api_key' in error_msg.lower():
                    st.error("🤖 Error de configuración de API Key.")
                else:
                    st.error(f"🤖 Error: {error_msg}")

            st.markdown("---")

            if peers:
                st.subheader(f"🔬 Análisis Comparativo con Competidores ({idioma})")
                
                with st.spinner("Obteniendo información de competidores..."):
                    peers_insights = {}
                    for p in peers[:5]:
                        try:
                            p_info = get_complete_info(p)
                            peers_insights[p] = p_info.get('longBusinessSummary', '')[:200]
                        except:
                            continue
                
                peers_data = []
                for p in peers[:5]:
                    try:
                        p_info = get_complete_info(p)
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
                    error_msg = st.session_state.get('ai_error', '')
                    
                    if 'cuota' in error_msg.lower() or 'quota' in error_msg.lower():
                        st.info("🤖 Límite de requests alcanzado. Intenta más tarde.")
                    elif 'api_key' in error_msg.lower():
                        st.error("🤖 Error de configuración de API Key.")
                    else:
                        st.error(f"🤖 Error: {error_msg}")

        else:
            st.info("🤖 Análisis de IA no disponible actualmente.")

        st.markdown("---")
        st.warning("⚠️ Esto no es recomendación financiera. Solo fines educativos.")

        st.markdown("""
        <div style='text-align:center; color:gray; font-size:11px; margin-top: 40px;'>
        📊 <b>Fuentes:</b> Yahoo Finance, Finviz & Banxico | 🤖 <b>IA:</b> Gemini 2.5 Flash<br>
        🎓 Ingeniería Financiera | 💻 Versión 5.0 - Web Scraping | ⚖️ Solo para uso educativo
        </div>
        """, unsafe_allow_html=True)
