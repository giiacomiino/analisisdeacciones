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
st.set_page_config(page_title="Análisis Integral de Acciones", layout="wide", initial_sidebar_state="expanded")
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Inicializar session_state
if "ticker" not in st.session_state:
    st.session_state["ticker"] = None
if "analizar" not in st.session_state:
    st.session_state["analizar"] = False

# -----------------------------
# FUNCIONES
# -----------------------------
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


def calcular_rendimientos_riesgos(ticker, indice_ticker, periodos_config, periodo_historico):
    resultados = []
    datos_ticker = yf.download(ticker, period=periodo_historico, interval="1d", progress=False)
    datos_indice = yf.download(indice_ticker, period=periodo_historico, interval="1d", progress=False)

    def extraer_precios(datos):
        if isinstance(datos.columns, pd.MultiIndex):
            if "Adj Close" in datos.columns.get_level_values(0):
                return datos["Adj Close"].iloc[:, 0].astype(float)
            return datos["Close"].iloc[:, 0].astype(float)
        else:
            if "Adj Close" in datos.columns:
                return datos["Adj Close"].astype(float)
            return datos["Close"].astype(float)

    precios_ticker = extraer_precios(datos_ticker).dropna()
    precios_indice = extraer_precios(datos_indice).dropna()

    precios_ticker, precios_indice = precios_ticker.align(precios_indice, join="inner")
    retornos_ticker = precios_ticker.pct_change().dropna()
    retornos_indice = precios_indice.pct_change().dropna()

    for p in periodos_config:
        try:
            if p['tipo'] == "YTD":
                inicio = datetime(datetime.now().year, 1, 1)
                pp = precios_ticker[precios_ticker.index >= inicio]
                rp = retornos_ticker[retornos_ticker.index >= inicio]
                rip = retornos_indice[retornos_indice.index >= inicio]
            else:
                dias = {"Años": 252, "Meses": 21, "Semanas": 5, "Días": 1, "Trimestres": 63}[p['tipo']] * p['cantidad']
                pp = precios_ticker.tail(dias)
                rp = retornos_ticker.tail(dias)
                rip = retornos_indice.tail(dias)

            if len(pp) < 10:
                resultados.append({
                    "Periodo": p['nombre'], "Rendimiento (%)": "N/A", "Volatilidad (%)": "N/A",
                    "Beta": "N/A", "Alpha (%)": "N/A", "Sharpe Ratio": "N/A"
                })
                continue

            rend = ((pp.iloc[-1] / pp.iloc[0]) - 1) * 100
            vol = rp.std() * np.sqrt(252) * 100
            beta = np.cov(rp, rip)[0, 1] / np.var(rip) if np.var(rip) != 0 else 0
            rend_ind = ((precios_indice[precios_indice.index.isin(pp.index)].iloc[-1] /
                         precios_indice[precios_indice.index.isin(pp.index)].iloc[0]) - 1) * 100
            alpha = rend - (beta * rend_ind)
            sharpe = (rp.mean() * 252 / rp.std()) if rp.std() != 0 else 0

            resultados.append({
                "Periodo": p['nombre'],
                "Rendimiento (%)": f"{rend:.2f}%",
                "Volatilidad (%)": f"{vol:.2f}%",
                "Beta": f"{beta:.2f}",
                "Alpha (%)": f"{alpha:.2f}%",
                "Sharpe Ratio": f"{sharpe:.2f}"
            })
        except:
            resultados.append({
                "Periodo": p['nombre'],
                "Rendimiento (%)": "Error", "Volatilidad (%)": "Error",
                "Beta": "Error", "Alpha (%)": "Error",
                "Sharpe Ratio": "Error"
            })

    return pd.DataFrame(resultados)


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    
    periodo_historico = st.selectbox(
        "📅 Periodo histórico:",
        ["1y", "3y", "5y", "10y", "max"],
        index=2,
    )
    
    st.divider()
    
    usar_predeterminados = st.checkbox("Usar periodos predeterminados", value=True)
    
    if not usar_predeterminados:
        num_periodos = st.number_input("Periodos personalizados:", 1, 10, 3)
        periodos_personalizados = []
        for i in range(num_periodos):
            with st.expander(f"Periodo {i+1}"):
                tipo = st.selectbox(f"Tipo:", ["Años", "Meses", "Trimestres", "Semanas", "Días", "YTD"], key=f"t{i}")
                cant = st.number_input(f"Cantidad:", 1, 120, 1, key=f"c{i}") if tipo != "YTD" else 0
                nom = st.text_input(f"Nombre:", f"{cant}{tipo[0]}" if tipo != "YTD" else "YTD", key=f"n{i}")
                periodos_personalizados.append({'nombre': nom, 'tipo': tipo, 'cantidad': cant})
    else:
        periodos_personalizados = [
            {'nombre': '3M', 'tipo': 'Meses', 'cantidad': 3},
            {'nombre': '6M', 'tipo': 'Meses', 'cantidad': 6},
            {'nombre': 'YTD', 'tipo': 'YTD', 'cantidad': 0},
            {'nombre': '1Y', 'tipo': 'Años', 'cantidad': 1},
            {'nombre': '3Y', 'tipo': 'Años', 'cantidad': 3},
        ]
    
    st.divider()
    st.markdown("**v3.5** | Ingeniería Financiera")

# -----------------------------
# ENCABEZADO PRINCIPAL
# -----------------------------
st.title("📊 Análisis Integral de Acciones")
st.caption("Análisis profesional con Yahoo Finance, Finviz y Gemini AI")


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
    
    col_ticker, col_reset = st.columns([5,1])
    with col_ticker:
        st.info(f"**Analizando:** {ticker_final}")
    with col_reset:
        if st.button("🔄 Nueva búsqueda"):
            st.session_state["ticker"] = None
            st.session_state["analizar"] = False
            st.rerun()

    st.markdown("---")

    # FILTROS DE CONFIGURACIÓN
    col1, col2 = st.columns(2)

    with col1:
        indices_dict = {
            "S&P 500": "^GSPC",
            "NASDAQ 100": "^NDX",
            "Dow Jones": "^DJI",
            "Russell 2000": "^RUT"
        }
        indice_select = st.selectbox("📈 Comparar contra:", list(indices_dict.keys()), key="indice_sel")

    with col2:
        idioma = st.selectbox(
            "🌐 Idioma del análisis:",
            ["Inglés", "Español", "Francés", "Alemán", "Italiano", "Portugués"],
            key="idioma_sel"
        )

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
            st.markdown("---")
            st.subheader("✨ Financial Insights (Yahoo Finance)")
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
        st.subheader("🏢 Información General")
        col1, col2 = st.columns(2)
        col1.markdown(f"**Nombre:** {info.get('longName', 'N/A')}")
        col1.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        col1.markdown(f"**Industria:** {info.get('industry', 'N/A')}")
        col2.markdown(f"**País:** {info.get('country', 'N/A')}")
        col2.markdown(f"**Empleados:** {info.get('fullTimeEmployees', 'N/A')}")

        desc = info.get('longBusinessSummary', 'Descripción no disponible.')
        with st.spinner(f"Traduciendo a {idioma}..."):
            desc_trad = traducir_descripcion(desc, idioma)
        st.markdown(f"**Descripción ({idioma}):** {desc_trad}")

        st.divider()


        # ==============================
        # KPIs
        # ==============================
        st.subheader("💡 KPIs Clave")
        kpis = {
            "Beta": info.get("beta", "N/A"),
            "P/E": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "ROE": f"{info.get('returnOnEquity', 0)*100:.1f}%" if info.get("returnOnEquity") else "N/A",
            "Gross Margin": f"{info.get('grossMargins', 0)*100:.1f}%" if info.get("grossMargins") else "N/A",
            "Profit Margin": f"{info.get('profitMargins', 0)*100:.1f}%" if info.get("profitMargins") else "N/A",
            "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get("dividendYield") else "—",
            "Market Cap": f"{info.get('marketCap', 0)/1e9:,.1f} B" if info.get("marketCap") else "N/A",
            "Revenue": f"{info.get('totalRevenue', 0)/1e9:,.1f} B" if info.get("totalRevenue") else "N/A",
            "Net Income": f"{info.get('netIncomeToCommon', 0)/1e9:,.1f} B" if info.get("netIncomeToCommon") else "N/A",
        }

        colores = ["#22313F", "#2C3E50", "#34495E", "#3A539B", "#1E8BC3", "#26A65B", "#8E44AD", "#C0392B", "#F39C12"]
        st.markdown("""<style>
        .kpi-bubble {border-radius: 18px; padding: 18px; text-align: center; color: white; 
                    box-shadow: 0 3px 10px rgba(0,0,0,0.3); margin-bottom: 18px;}
        .kpi-title {font-size: 13px; color: #BDC3C7; margin-bottom: 4px; font-weight: 500;}
        .kpi-value {font-size: 20px; font-weight: bold; color: white;}
        </style>""", unsafe_allow_html=True)

        cols = st.columns(3)
        for i, (k, v) in enumerate(kpis.items()):
            with cols[i % 3]:
                st.markdown(
                    f"""<div class="kpi-bubble" style="background:{colores[i % len(colores)]}">
                            <div class="kpi-title">{k}</div>
                            <div class="kpi-value">{v}</div>
                        </div>""",
                    unsafe_allow_html=True
                )

        st.divider()


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

                st.divider()


        # ==============================
        # GRÁFICO DE VELAS
        # ==============================
        st.subheader("📊 Gráfico de Velas")
        datos = yf.download(ticker_final, period="1y", interval="1d", progress=False)

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
                              title=f"Precio Histórico - {ticker_final}")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()


        # ==============================
        # COMPARACIÓN CONTRA ÍNDICE
        # ==============================
        st.subheader("📈 Rendimiento Comparativo")

        try:
            datos_ticker = yf.download(ticker_final, period="1y", interval="1d", progress=False)
            indice_t = indices_dict[indice_select]
            datos_indice = yf.download(indice_t, period="1y", interval="1d", progress=False)

            if isinstance(datos_ticker.columns, pd.MultiIndex):
                precios_ticker = datos_ticker["Adj Close"].iloc[:, 0]
            else:
                precios_ticker = datos_ticker["Adj Close"]

            if isinstance(datos_indice.columns, pd.MultiIndex):
                precios_indice = datos_indice["Adj Close"].iloc[:, 0]
            else:
                precios_indice = datos_indice["Adj Close"]

            precios_ticker, precios_indice = precios_ticker.align(precios_indice, join="inner")

            rendimiento_ticker = (precios_ticker / precios_ticker.iloc[0]) * 100
            rendimiento_indice = (precios_indice / precios_indice.iloc[0]) * 100

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=rendimiento_ticker.index, y=rendimiento_ticker.values,
                mode="lines", name=ticker_final, line=dict(color="#1E8BC3", width=3)
            ))
            fig_comp.add_trace(go.Scatter(
                x=rendimiento_indice.index, y=rendimiento_indice.values,
                mode="lines", name=indice_select, line=dict(color="#E67E22", width=3, dash="dot")
            ))

            fig_comp.update_layout(title=f"{ticker_final} vs {indice_select}", template="plotly_white", height=500)
            st.plotly_chart(fig_comp, use_container_width=True)

        except Exception as e:
            st.warning(f"No fue posible generar la comparativa: {e}")

        st.divider()


        # ==============================
        # RENDIMIENTOS Y RIESGOS
        # ==============================
        st.subheader("📊 Rendimientos y Riesgos")
        df_analisis = calcular_rendimientos_riesgos(
            ticker_final, indice_t, periodos_personalizados, periodo_historico
        )
        st.dataframe(df_analisis, use_container_width=True, hide_index=True)

        st.divider()


        # ==============================
        # INCOME STATEMENT
        # ==============================
        st.subheader("📘 Income Statement")
        df_income = obtener_income_yahoo(ticker_final)

        if not df_income.empty:
            st.dataframe(df_income, use_container_width=True, hide_index=True)

        st.divider()


        # ==============================
        # ANÁLISIS INDIVIDUAL CON GEMINI
        # ==============================
        st.subheader(f"🧠 Análisis Individual con Gemini AI ({idioma})")

        modelo = genai.GenerativeModel("gemini-2.5-flash")

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
            analisis_individual = modelo.generate_content(prompt_individual).text.strip()

        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 25px; border-radius: 12px; color: white; margin-bottom: 20px;'>
                {analisis_individual.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()


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

Da la respuesta en formato plano, sin asteriscos ni formato markdown.
"""

            with st.spinner("Generando análisis comparativo con peers..."):
                analisis_comparativo = modelo.generate_content(prompt_comparativo).text.strip()

            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            padding: 25px; border-radius: 12px; color: white;'>
                    {analisis_comparativo.replace(chr(10), "<br>")}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.warning("⚠️ Esto no es recomendación financiera. Solo fines educativos.")

        st.divider()


        # ==============================
        # FOOTER
        # ==============================
        st.markdown("""
        <div style='text-align:center; color:gray; font-size:11px;'>
        📊 <b>Fuentes:</b> Yahoo Finance & Finviz | 🤖 <b>IA:</b> Gemini 2.5 Flash<br>
        🎓 Ingeniería Financiera | 💻 Versión 3.5 | ⚖️ Solo para uso educativo
        </div>
        """, unsafe_allow_html=True)