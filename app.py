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
def obtener_tasa_libre_riesgo():
    """Obtiene la tasa CETES 28 desde Banxico como proxy de tasa libre de riesgo."""
    try:
        url = "https://www.banxico.org.mx/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        cetes_div = soup.find("div", {"id": "dv_singleCetes28"})
        if cetes_div:
            valor_span = cetes_div.find("span", class_="valor")
            if valor_span:
                tasa = float(valor_span.get_text(strip=True))
                return tasa / 100
        
        return 0.07
    except:
        return 0.07


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


def obtener_financial_insights_peers(peers_tickers):
    insights_dict = {}
    for t in peers_tickers:
        insight = obtener_financial_insights_yf(t)
        if insight:
            insights_dict[t] = insight
    return insights_dict


def obtener_income_yahoo(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.financials
        claves = ["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income", "Net Income", "EBIT", "EBITDA"]
        if df.empty:
            return pd.DataFrame()
        df = df.loc[df.index.intersection(claves)].reindex(claves).fillna(0).astype(float).T
        df.index = [i.strftime("%Y-%m-%d") for i in df.index]
        return df
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


def calcular_metricas_periodo(ticker, indice_ticker, periodo_dias, tasa_libre_riesgo):
    """Calcula métricas de rendimiento y riesgo para un periodo específico."""
    try:
        datos_ticker = yf.download(ticker, period="5y", interval="1d", progress=False)
        datos_indice = yf.download(indice_ticker, period="5y", interval="1d", progress=False)

        precios_ticker = extraer_precios_columna(datos_ticker).dropna()
        precios_indice = extraer_precios_columna(datos_indice).dropna()

        precios_ticker, precios_indice = precios_ticker.align(precios_indice, join="inner")
        
        if periodo_dias == "YTD":
            inicio = datetime(datetime.now().year, 1, 1)
            pp = precios_ticker[precios_ticker.index >= inicio]
            pi = precios_indice[precios_indice.index >= inicio]
        elif periodo_dias == "max":
            pp = precios_ticker
            pi = precios_indice
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

if not GEMINI_DISPONIBLE:
    st.warning("⚠️ **Funcionalidad limitada:** El servicio de IA (Gemini) no está disponible actualmente debido a límite de requests. La app funcionará con todas las métricas y gráficos, pero sin análisis de IA ni traducciones.")

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
            st.markdown('<div class="section-header">✨ Insights Financieros Clave</div>', unsafe_allow_html=True)
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
        # INFORMACIÓN GENERAL
        # ==============================
        st.markdown("""
            <style>
            .info-card {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                padding: 25px;
                border-radius: 15px;
                color: white;
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                margin-bottom: 15px;
                transition: transform 0.3s ease;
            }
            .info-card:hover {
                transform: translateY(-5px);
            }
            .info-label {
                font-size: 12px;
                opacity: 0.8;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 5px;
            }
            .info-value {
                font-size: 24px;
                font-weight: bold;
                color: #ffffff;
            }
            .section-header {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 20px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">🏢 Perfil Corporativo</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="info-card">
                    <div class="info-label">🏢 Empresa</div>
                    <div class="info-value">{info.get('longName', 'N/A')[:30]}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="info-card">
                    <div class="info-label">🏭 Sector</div>
                    <div class="info-value">{info.get('sector', 'N/A')}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="info-card">
                    <div class="info-label">🌍 País</div>
                    <div class="info-value">{info.get('country', 'N/A')}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            empleados = f"{info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else "N/A"
            st.markdown(f"""
                <div class="info-card">
                    <div class="info-label">👥 Empleados</div>
                    <div class="info-value">{empleados}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="info-card">
                <div class="info-label">🔧 Industria</div>
                <div class="info-value">{info.get('industry', 'N/A')}</div>
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
        # KPIs - MÉTRICAS BURSÁTILES (AMPLIADAS)
        # ==============================
        st.markdown('<div class="section-header">📈 Radiografía Bursátil</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <style>
            .kpi-card {
                background: linear-gradient(135deg, #134E5E 0%, #71B280 100%);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: white;
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                margin-bottom: 10px;
                transition: all 0.3s ease;
                border: 2px solid transparent;
                position: relative;
            }
            .kpi-card:hover {
                transform: translateY(-8px);
                border: 2px solid #71B280;
                box-shadow: 0 10px 30px rgba(113, 178, 128, 0.4);
            }
            .kpi-card:hover .tooltip {
                visibility: visible;
                opacity: 1;
            }
            .kpi-label {
                font-size: 13px;
                opacity: 0.9;
                margin-bottom: 8px;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            .kpi-value {
                font-size: 28px;
                font-weight: bold;
                color: white;
            }
            .tooltip {
                visibility: hidden;
                width: 220px;
                background-color: #333;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -110px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 11px;
                line-height: 1.4;
            }
            .tooltip::after {
                content: "";
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: #333 transparent transparent transparent;
            }
            </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            market_cap = f"${info.get('marketCap', 0)/1e9:,.1f}B" if info.get('marketCap') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">Capitalización de mercado: valor total de todas las acciones en circulación</div>
                    <div class="kpi-label">💰 Market Cap</div>
                    <div class="kpi-value">{market_cap}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pe = f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">Price/Earnings: relación precio-ganancia. Indica cuánto paga el mercado por cada dólar de ganancia</div>
                    <div class="kpi-label">📊 P/E Ratio</div>
                    <div class="kpi-value">{pe}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            eps = f"${info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">Earnings Per Share: ganancia neta dividida entre el número de acciones en circulación</div>
                    <div class="kpi-label">💵 EPS</div>
                    <div class="kpi-value">{eps}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            beta = f"{info.get('beta', 0):.2f}" if info.get('beta') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">Beta: mide la volatilidad relativa vs el mercado. >1 más volátil, <1 menos volátil</div>
                    <div class="kpi-label">📉 Beta</div>
                    <div class="kpi-value">{beta}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            div_yield = f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "—"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">Dividend Yield: rendimiento anual por dividendos como % del precio de la acción</div>
                    <div class="kpi-label">💸 Div. Yield</div>
                    <div class="kpi-value">{div_yield}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col6:
            pb = f"{info.get('priceToBook', 0):.2f}" if info.get('priceToBook') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">Price/Book: relación entre precio de mercado y valor en libros. Indica si está sobre o infravalorada</div>
                    <div class="kpi-label">📖 P/B Ratio</div>
                    <div class="kpi-value">{pb}</div>
                </div>
            """, unsafe_allow_html=True)

        # Segunda fila de métricas bursátiles
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            peg = f"{info.get('pegRatio', 0):.2f}" if info.get('pegRatio') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">PEG Ratio: P/E ajustado por crecimiento. <1 puede indicar infravaloración</div>
                    <div class="kpi-label">🎯 PEG Ratio</div>
                    <div class="kpi-value">{peg}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ps = f"{info.get('priceToSalesTrailing12Months', 0):.2f}" if info.get('priceToSalesTrailing12Months') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">Price/Sales: capitalización de mercado dividida entre ingresos totales</div>
                    <div class="kpi-label">💼 P/S Ratio</div>
                    <div class="kpi-value">{ps}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            target_mean = f"${info.get('targetMeanPrice', 0):.2f}" if info.get('targetMeanPrice') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">Target Price: precio objetivo promedio según analistas</div>
                    <div class="kpi-label">🎯 Target Price</div>
                    <div class="kpi-value">{target_mean}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            current_price = f"${info.get('currentPrice', 0):.2f}" if info.get('currentPrice') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">Current Price: precio actual de la acción en el mercado</div>
                    <div class="kpi-label">💲 Precio Actual</div>
                    <div class="kpi-value">{current_price}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            week_high = f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">52-Week High: precio máximo alcanzado en los últimos 52 semanas</div>
                    <div class="kpi-label">📈 52w High</div>
                    <div class="kpi-value">{week_high}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col6:
            week_low = f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else "N/A"
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="tooltip">52-Week Low: precio mínimo alcanzado en los últimos 52 semanas</div>
                    <div class="kpi-label">📉 52w Low</div>
                    <div class="kpi-value">{week_low}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==============================
        # KPIs - MÉTRICAS CORPORATIVAS (AMPLIADAS)
        # ==============================
        st.markdown('<div class="section-header">🏦 Análisis de Rentabilidad</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <style>
            .corp-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: white;
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                margin-bottom: 10px;
                transition: all 0.3s ease;
                border: 2px solid transparent;
                position: relative;
            }
            .corp-card:hover {
                transform: translateY(-8px);
                border: 2px solid #764ba2;
                box-shadow: 0 10px 30px rgba(118, 75, 162, 0.4);
            }
            .corp-card:hover .tooltip {
                visibility: visible;
                opacity: 1;
            }
            .corp-label {
                font-size: 13px;
                opacity: 0.9;
                margin-bottom: 8px;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            .corp-value {
                font-size: 28px;
                font-weight: bold;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            roe = f"{info.get('returnOnEquity', 0)*100:.1f}%" if info.get('returnOnEquity') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Return on Equity: rentabilidad sobre el capital. Mide eficiencia en generar ganancias con el capital de accionistas</div>
                    <div class="corp-label">📈 ROE</div>
                    <div class="corp-value">{roe}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            roa = f"{info.get('returnOnAssets', 0)*100:.1f}%" if info.get('returnOnAssets') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Return on Assets: rentabilidad sobre activos totales. Mide eficiencia en uso de activos</div>
                    <div class="corp-label">💼 ROA</div>
                    <div class="corp-value">{roa}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            gross_margin = f"{info.get('grossMargins', 0)*100:.1f}%" if info.get('grossMargins') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Gross Margin: margen bruto. (Ingresos - Costo de ventas) / Ingresos. Mayor = mejor poder de fijación de precios</div>
                    <div class="corp-label">📊 Gross Margin</div>
                    <div class="corp-value">{gross_margin}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            profit_margin = f"{info.get('profitMargins', 0)*100:.1f}%" if info.get('profitMargins') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Profit Margin: margen neto. Ganancia neta / Ingresos. Indica cuánto queda después de todos los gastos</div>
                    <div class="corp-label">💹 Profit Margin</div>
                    <div class="corp-value">{profit_margin}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            operating_margin = f"{info.get('operatingMargins', 0)*100:.1f}%" if info.get('operatingMargins') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Operating Margin: margen operativo. Ganancia operativa / Ingresos. Eficiencia operativa antes de intereses e impuestos</div>
                    <div class="corp-label">⚙️ Operating Margin</div>
                    <div class="corp-value">{operating_margin}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col6:
            ebitda_margin = f"{info.get('ebitdaMargins', 0)*100:.1f}%" if info.get('ebitdaMargins') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">EBITDA Margin: EBITDA / Ingresos. Rentabilidad operativa antes de depreciación y amortización</div>
                    <div class="corp-label">📊 EBITDA Margin</div>
                    <div class="corp-value">{ebitda_margin}</div>
                </div>
            """, unsafe_allow_html=True)

        # Segunda fila de métricas corporativas
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            revenue = f"${info.get('totalRevenue', 0)/1e9:,.1f}B" if info.get('totalRevenue') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Revenue TTM: ingresos totales de los últimos 12 meses</div>
                    <div class="corp-label">💰 Revenue TTM</div>
                    <div class="corp-value">{revenue}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            net_income = f"${info.get('netIncomeToCommon', 0)/1e9:,.1f}B" if info.get('netIncomeToCommon') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Net Income: ganancia neta después de todos los gastos e impuestos</div>
                    <div class="corp-label">💵 Net Income</div>
                    <div class="corp-value">{net_income}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            ebitda = f"${info.get('ebitda', 0)/1e9:,.1f}B" if info.get('ebitda') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">EBITDA: Earnings Before Interest, Taxes, Depreciation & Amortization. Mide rentabilidad operativa</div>
                    <div class="corp-label">📈 EBITDA</div>
                    <div class="corp-value">{ebitda}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            debt_equity = f"{info.get('debtToEquity', 0)/100:.2f}" if info.get('debtToEquity') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Debt/Equity: apalancamiento financiero. Deuda total / Capital. >2 puede indicar alto riesgo</div>
                    <div class="corp-label">⚖️ Debt/Equity</div>
                    <div class="corp-value">{debt_equity}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            current_ratio = f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Current Ratio: liquidez. Activos corrientes / Pasivos corrientes. >1 indica buena liquidez</div>
                    <div class="corp-label">💧 Current Ratio</div>
                    <div class="corp-value">{current_ratio}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col6:
            quick_ratio = f"{info.get('quickRatio', 0):.2f}" if info.get('quickRatio') else "N/A"
            st.markdown(f"""
                <div class="corp-card">
                    <div class="tooltip">Quick Ratio: prueba ácida. (Activos corrientes - Inventarios) / Pasivos corrientes. Liquidez inmediata</div>
                    <div class="corp-label">⚡ Quick Ratio</div>
                    <div class="corp-value">{quick_ratio}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")


        # ==============================
        # COMPARACIÓN CON PEERS
        # ==============================
        st.markdown('<div class="section-header">🔍 Comparativa Competitiva</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-header">📊 Evolución del Precio</div>', unsafe_allow_html=True)
        
        periodo_velas = st.selectbox(
            "Selecciona el periodo:",
            ["3 meses", "6 meses", "YTD", "1 año", "3 años", "5 años", "max"],
            index=3,
            key="periodo_velas"
        )
        
        periodo_map = {
            "3 meses": "3mo",
            "6 meses": "6mo",
            "YTD": "ytd",
            "1 año": "1y",
            "3 años": "3y",
            "5 años": "5y",
            "max": "max"
        }
        
        datos = yf.download(ticker_final, period=periodo_map[periodo_velas], interval="1d", progress=False)

        if not datos.empty:
            if isinstance(datos.columns, pd.MultiIndex):
                open_col = datos["Open"].iloc[:, 0]
                high_col = datos["High"].iloc[:, 0]
                low_col = datos["Low"].iloc[:, 0]
                close_col = datos["Close"].iloc[:, 0]
            else:
                open_col = datos["Open"]
                high_col = datos["High"]
                low_col = datos["Low"]
                close_col = datos["Close"]

            fig = go.Figure(go.Candlestick(
                x=datos.index,
                open=open_col,
                high=high_col,
                low=low_col,
                close=close_col,
                increasing_line_color="#26A65B",
                decreasing_line_color="#C0392B"
            ))

            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                              title=f"Precio Histórico - {ticker_final} ({periodo_velas})")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")


        # ==============================
        # COMPARACIÓN CONTRA ÍNDICE (BASE 0)
        # ==============================
        st.markdown('<div class="section-header">📈 Performance vs Índice de Referencia</div>', unsafe_allow_html=True)
        
        periodo_comp_indice = st.selectbox(
            "Selecciona el periodo:",
            ["3 meses", "6 meses", "YTD", "1 año", "3 años", "5 años", "max"],
            index=3,
            key="periodo_comp_indice"
        )

        try:
            datos_ticker = yf.download(ticker_final, period=periodo_map[periodo_comp_indice], interval="1d", progress=False)
            indice_t = indices_dict[indice_select]
            datos_indice = yf.download(indice_t, period=periodo_map[periodo_comp_indice], interval="1d", progress=False)

            precios_ticker = extraer_precios_columna(datos_ticker)
            precios_indice = extraer_precios_columna(datos_indice)

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
                title=f"{ticker_final} vs {indice_select} (Base 0) - {periodo_comp_indice}", 
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
            st.markdown('<div class="section-header">📊 Battle Royale: Performance vs Competidores</div>', unsafe_allow_html=True)
            
            periodo_peers = st.selectbox(
                "Selecciona el periodo:",
                ["3 meses", "6 meses", "YTD", "1 año", "3 años", "5 años", "max"],
                index=3,
                key="periodo_peers"
            )

            try:
                fig_peers = go.Figure()
                
                datos_main = yf.download(ticker_final, period=periodo_map[periodo_peers], interval="1d", progress=False)
                precios_main = extraer_precios_columna(datos_main)
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
                        datos_peer = yf.download(peer, period=periodo_map[periodo_peers], interval="1d", progress=False)
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

                fig_peers.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                fig_peers.update_layout(
                    title=f"Rendimiento Comparativo: {ticker_final} vs Peers (Base 0) - {periodo_peers}",
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
        st.markdown('<div class="section-header">📊 Métricas de Riesgo y Retorno</div>', unsafe_allow_html=True)
        
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
                    position: relative;
                }
                .metric-bubble:hover {
                    transform: translateY(-5px);
                }
                .metric-bubble:hover .tooltip {
                    visibility: visible;
                    opacity: 1;
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
                    <div class="tooltip">Rendimiento total en el periodo seleccionado</div>
                    <div class="metric-title">Rendimiento</div>
                    <div class="metric-value">{metricas['rendimiento']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            col2.markdown(f"""
                <div class="metric-bubble">
                    <div class="tooltip">Volatilidad anualizada: desviación estándar de los retornos</div>
                    <div class="metric-title">Volatilidad</div>
                    <div class="metric-value">{metricas['volatilidad']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            col3.markdown(f"""
                <div class="metric-bubble">
                    <div class="tooltip">Beta: sensibilidad al índice de referencia. >1 más volátil que el mercado</div>
                    <div class="metric-title">Beta</div>
                    <div class="metric-value">{metricas['beta']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            col4.markdown(f"""
                <div class="metric-bubble">
                    <div class="tooltip">Alpha: rendimiento en exceso vs el índice ajustado por beta</div>
                    <div class="metric-title">Alpha</div>
                    <div class="metric-value">{metricas['alpha']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            col5.markdown(f"""
                <div class="metric-bubble">
                    <div class="tooltip">Sharpe Ratio: rendimiento ajustado por riesgo. >1 es bueno, >2 muy bueno</div>
                    <div class="metric-title">Sharpe Ratio</div>
                    <div class="metric-value">{metricas['sharpe']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"📊 Métricas calculadas para el periodo: **{periodo_sel}** vs **{indice_select}** | 📈 Tasa libre de riesgo (CETES 28): **{tasa_libre_riesgo*100:.2f}%**")
        else:
            st.warning("No se pudieron calcular las métricas para este periodo.")

        st.markdown("---")


        # ==============================
        # INCOME STATEMENT - FORMATO VISUAL
        # ==============================
        st.markdown('<div class="section-header">📘 Estado de Resultados</div>', unsafe_allow_html=True)
        df_income = obtener_income_yahoo(ticker_final)

        if not df_income.empty:
            st.markdown("""
                <style>
                .income-card {
                    background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
                    padding: 20px;
                    border-radius: 12px;
                    color: white;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
                    margin-bottom: 15px;
                    transition: transform 0.3s ease;
                }
                .income-card:hover {
                    transform: translateY(-3px);
                }
                .income-metric {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                .income-metric:last-child {
                    border-bottom: none;
                }
                .income-label {
                    font-size: 15px;
                    font-weight: 500;
                    opacity: 0.9;
                }
                .income-value {
                    font-size: 16px;
                    font-weight: bold;
                    text-align: right;
                }
                .income-header {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #71B280;
                    text-align: center;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Crear columnas para cada periodo
            periodos = df_income.index.tolist()
            num_cols = len(periodos)
            cols = st.columns(num_cols)
            
            metricas = df_income.columns.tolist()
            
            for i, periodo in enumerate(periodos):
                with cols[i]:
                    card_html = f'<div class="income-card"><div class="income-header">{periodo}</div>'
                    
                    for metrica in metricas:
                        valor = df_income.loc[periodo, metrica]
                        
                        # Formatear valor
                        if abs(valor) >= 1e9:
                            valor_fmt = f"${valor/1e9:,.2f}B"
                        elif abs(valor) >= 1e6:
                            valor_fmt = f"${valor/1e6:,.1f}M"
                        else:
                            valor_fmt = f"${valor:,.0f}"
                        
                        # Color según el tipo de métrica
                        if metrica in ["Total Revenue", "Gross Profit", "EBITDA", "Net Income"]:
                            color = "#26A65B" if valor >= 0 else "#C0392B"
                        else:
                            color = "#E67E22"
                        
                        card_html += f'''
                        <div class="income-metric">
                            <div class="income-label">{metrica}</div>
                            <div class="income-value" style="color: {color};">{valor_fmt}</div>
                        </div>
                        '''
                    
                    card_html += '</div>'
                    st.markdown(card_html, unsafe_allow_html=True)
        
        else:
            st.info("📊 Estado de Resultados no disponible para este ticker")

        st.markdown("---")


        # ==============================
        # ANÁLISIS INDIVIDUAL CON GEMINI
        # ==============================
        if GEMINI_DISPONIBLE:
            st.markdown(f'<div class="section-header">🧠 Análisis IA Individual ({idioma})</div>', unsafe_allow_html=True)

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
1. Recomendación de inversión (Comprar/Mantener/Vender)
2. Fortalezas clave
3. Riesgos principales
4. Valoración actual
5. Perspectiva a corto plazo (3-6 meses)
6. Perspectiva a largo plazo (1-3 años)

Da la respuesta en formato plano, sin asteriscos ni formato markdown.
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
                st.markdown(f'<div class="section-header">🔬 Análisis IA Comparativo ({idioma})</div>', unsafe_allow_html=True)
                
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

Da la respuesta en formato plano, sin asteriscos ni formato markdown.
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

        st.markdown("""
        <div style='text-align:center; color:gray; font-size:11px; margin-top: 40px;'>
        📊 <b>Fuentes:</b> Yahoo Finance, Finviz & Banxico | 🤖 <b>IA:</b> Gemini 2.5 Flash<br>
        🎓 Ingeniería Financiera | 💻 Versión 5.0 | ⚖️ Solo para uso educativo
        </div>
        """, unsafe_allow_html=True)