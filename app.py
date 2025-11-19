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
        if "Adj Close" in datos.columns:
            return datos["Adj Close"].iloc[:, 0].astype(float) if isinstance(datos.columns, pd.MultiIndex) else datos["Adj Close"].astype(float)
        return datos["Close"].iloc[:, 0].astype(float) if isinstance(datos.columns, pd.MultiIndex) else datos["Close"].astype(float)

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
                "Beta": "Error", "Alpha (%)": "Error", "Sharpe Ratio": "Error"
            })

    return pd.DataFrame(resultados)
# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    
    periodo_historico = st.selectbox(
        "📅 Periodo histórico:",
        ["1Y", "3Y", "5Y", "10Y", "MAX"],
        index=2,
        help="Selecciona el periodo de datos históricos a descargar"
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
    st.markdown("**v3.3** | Ingeniería Financiera")


# -----------------------------
# ENCABEZADO PRINCIPAL
# -----------------------------
st.title("📊 Análisis Integral de Acciones")
st.caption("Análisis profesional con Yahoo Finance, Finviz y Gemini AI")


# -----------------------------
# 🔍 SEARCHBAR INTELIGENTE — ANTES DE TODO
# -----------------------------
st.subheader("🔎 Buscar Empresa / Ticker")

busqueda = st.text_input(
    "Escribe el nombre de la empresa o el ticker:",
    placeholder="Ejemplo: Apple, Tesla, Amazon, Coca-Cola..."
)

resultados = []
ticker_seleccionado = None

if len(busqueda) >= 2:
    with st.spinner("Buscando empresas..."):
        resultados = buscar_empresas_detallado(busqueda)


# Mostrar tarjetas visuales
if resultados:
    st.write("### Resultados encontrados:")

    for item in resultados:
        col1, col2 = st.columns([1,4])

        with col1:
            if item["logo"]:
                st.image(item["logo"], width=60)
            else:
                st.write("🗂")

        with col2:
            st.markdown(f"""
                **{item['nombre']}**  
                `{item['ticker']}`  
                **Precio actual:** {item['precio']} USD  
                **País:** {item['pais']}
            """)

            # Botón para seleccionar la empresa
            if st.button(f"Seleccionar {item['ticker']}", key=item['ticker']):
                ticker_seleccionado = item["ticker"]
                st.session_state["ticker"] = ticker_seleccionado


# Si ya se guardó en sesión, úsalo
if "ticker" in st.session_state:
    ticker_final = st.session_state["ticker"]
else:
    ticker_final = "AAPL"   # valor inicial por defecto


# -----------------------------
# FILTROS ADICIONALES
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    indices_dict = {"S&P 500": "^GSPC", "NASDAQ 100": "^NDX", "Dow Jones": "^DJI", "Russell 2000": "^RUT"}
    indice_select = st.selectbox("📈 Comparar contra:", list(indices_dict.keys()))

with col2:
    idioma = st.selectbox("🌐 Idioma:", ["Inglés", "Español", "Francés", "Alemán", "Italiano", "Portugués"])


# -----------------------------
# BOTÓN PARA ANALIZAR
# -----------------------------
if st.button("🚀 Analizar", type="primary"):
    ticker = ticker_final  # usar el elegido
    try:
        ticker_info = yf.Ticker(ticker)
        info = ticker_info.info

        if not info:
            raise ValueError("No se pudo obtener información del ticker")
    except Exception as e:
        st.error(f"❌ No se pudo cargar la información del ticker: {e}")
        st.stop()


        # Aquí continúa el análisis completo...
        # ==============================
        # FINANCIAL INSIGHTS DE YAHOO FINANCE
        # ==============================
        financial_insights = obtener_financial_insights_yf(ticker)
        
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
                        Powered by Yahoo Finance AI
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
                st.markdown(f"""<div class="kpi-bubble" style="background:{colores[i % len(colores)]}">
                    <div class="kpi-title">{k}</div><div class="kpi-value">{v}</div></div>""", unsafe_allow_html=True)
        
        st.divider()
        
        # ==============================
        # COMPARACIÓN CON PEERS
        # ==============================
        st.subheader("🔍 Comparación con Competidores")
        peers = obtener_peers_finviz(ticker)
        df_comp = pd.DataFrame()
        
        if peers:
            df_comp = obtener_kpis_peers(ticker, peers)
            if not df_comp.empty:
                def highlight(row):
                    return ['background-color: #1E8BC3; color: white; font-weight: bold'] * len(row) if row['Ticker'] == ticker else [''] * len(row)
                st.dataframe(df_comp.style.apply(highlight, axis=1), use_container_width=True, hide_index=True, height=400)
                
                col1, col2 = st.columns(2)
                with col1:
                    df_pe = df_comp[df_comp['P/E'] != 'N/A'].copy()
                    df_pe['P/E_num'] = df_pe['P/E'].astype(float)
                    fig_pe = go.Figure(go.Bar(x=df_pe['Ticker'], y=df_pe['P/E_num'], 
                                              marker_color=['#1E8BC3' if t == ticker else '#34495E' for t in df_pe['Ticker']],
                                              text=df_pe['P/E_num'].round(2), textposition='outside'))
                    fig_pe.update_layout(title="P/E Ratio Comparativo", xaxis_title="Empresa", yaxis_title="P/E Ratio",
                                        template="plotly_white", height=400, showlegend=False)
                    st.plotly_chart(fig_pe, use_container_width=True)
                
                with col2:
                    df_roe = df_comp[df_comp['ROE (%)'] != 'N/A'].copy()
                    df_roe['ROE_num'] = df_roe['ROE (%)'].astype(float)
                    fig_roe = go.Figure(go.Bar(x=df_roe['Ticker'], y=df_roe['ROE_num'],
                                               marker_color=['#26A65B' if t == ticker else '#34495E' for t in df_roe['Ticker']],
                                               text=df_roe['ROE_num'].round(2), textposition='outside'))
                    fig_roe.update_layout(title="ROE (%) Comparativo", xaxis_title="Empresa", yaxis_title="ROE %",
                                         template="plotly_white", height=400, showlegend=False)
                    st.plotly_chart(fig_roe, use_container_width=True)
                
                # ==============================
                # GRÁFICO DE MARKET CAP
                # ==============================
                st.subheader("💰 Capitalización de Mercado Comparativa")
                df_mcap = df_comp[df_comp['Market Cap (B)'] != 'N/A'].copy()
                df_mcap['Market Cap_num'] = df_mcap['Market Cap (B)'].astype(float)
                df_mcap = df_mcap.sort_values('Market Cap_num', ascending=True)
                
                fig_mcap = go.Figure(go.Bar(
                    x=df_mcap['Market Cap_num'],
                    y=df_mcap['Ticker'],
                    orientation='h',
                    marker_color=['#E67E22' if t == ticker else '#95A5A6' for t in df_mcap['Ticker']],
                    text=df_mcap['Market Cap_num'].round(2),
                    textposition='outside'
                ))
                fig_mcap.update_layout(
                    title="Market Cap (Miles de Millones USD)",
                    xaxis_title="Market Cap (B)",
                    yaxis_title="Empresa",
                    template="plotly_white",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_mcap, use_container_width=True)
        
        st.divider()
        
        # ==============================
        # GRÁFICO DE VELAS
        # ==============================
        st.subheader("📊 Gráfico de Velas")
        datos = yf.download(ticker, period="1y", interval="1d", progress=False)
        if not datos.empty:
            def get_col(datos, col_name):
                if isinstance(datos.columns, pd.MultiIndex):
                    return datos[col_name].iloc[:, 0] if col_name in datos.columns.get_level_values(0) else datos.iloc[:, 0]
                return datos[col_name]
            
            fig = go.Figure(go.Candlestick(
                x=datos.index,
                open=get_col(datos, 'Open'),
                high=get_col(datos, 'High'),
                low=get_col(datos, 'Low'),
                close=get_col(datos, 'Close'),
                increasing_line_color='#26A65B',
                decreasing_line_color='#C0392B'
            ))
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, 
                             title=f"Precio Histórico - {ticker}", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # ==============================
        # COMPARACIÓN CONTRA ÍNDICE (BASE CERO)
        # ==============================
        st.subheader("📈 Rendimiento Histórico Comparativo (Base Cero)")
        
        try:
            datos_ticker = yf.download(ticker, period="1y", interval="1d", progress=False)
            indice_ticker = indices_dict[indice_select]
            datos_indice = yf.download(indice_ticker, period="1y", interval="1d", progress=False)
            
            def extraer_precios(datos):
                if "Adj Close" in datos.columns:
                    return datos["Adj Close"].iloc[:, 0].astype(float).dropna() if isinstance(datos.columns, pd.MultiIndex) else datos["Adj Close"].astype(float).dropna()
                return datos["Close"].iloc[:, 0].astype(float).dropna() if isinstance(datos.columns, pd.MultiIndex) else datos["Close"].astype(float).dropna()
            
            precios_ticker = extraer_precios(datos_ticker)
            precios_indice = extraer_precios(datos_indice)
            
            if len(precios_ticker) > 2 and len(precios_indice) > 2:
                precios_ticker_aligned, precios_indice_aligned = precios_ticker.align(precios_indice, join="inner")
                
                if len(precios_ticker_aligned) > 2 and len(precios_indice_aligned) > 2:
                    rendimiento_ticker = (precios_ticker_aligned / precios_ticker_aligned.iloc[0]) * 100
                    rendimiento_indice = (precios_indice_aligned / precios_indice_aligned.iloc[0]) * 100
                    
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(
                        x=rendimiento_ticker.index, y=rendimiento_ticker.values,
                        mode="lines", name=ticker, line=dict(color="#1E8BC3", width=3),
                        hovertemplate=f"{ticker}: %{{y:.2f}}<extra></extra>"
                    ))
                    fig_comp.add_trace(go.Scatter(
                        x=rendimiento_indice.index, y=rendimiento_indice.values,
                        mode="lines", name=indice_select, line=dict(color="#E67E22", width=3, dash="dot"),
                        hovertemplate=f"{indice_select}: %{{y:.2f}}<extra></extra>"
                    ))

                    fig_comp.update_layout(
                        title=f"Comparativa: {ticker} vs {indice_select} (Base 100)",
                        yaxis_title="Índice Base 100", xaxis_title="Fecha",
                        template="plotly_white", height=500, hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    diferencia = rendimiento_ticker.iloc[-1] - rendimiento_indice.iloc[-1]
                    if diferencia > 0:
                        st.success(f"🟢 **{ticker}** superó al {indice_select} en **{diferencia:.2f}** puntos porcentuales")
                    else:
                        st.error(f"🔴 **{ticker}** quedó por debajo del {indice_select} en **{abs(diferencia):.2f}** puntos porcentuales")

        except Exception as e:
            st.warning(f"No se pudo generar la comparativa: {e}")
        
        st.divider()

        # ==============================
        # RENDIMIENTOS Y RIESGOS
        # ==============================
        st.subheader("📊 Análisis de Rendimientos y Riesgos")
        st.caption(f"Periodo histórico: {periodo_historico}")
        
        indice_ticker = indices_dict[indice_select]
        df_analisis = calcular_rendimientos_riesgos(
            ticker, indice_ticker, periodos_personalizados, periodo_historico.lower()
        )
        st.dataframe(df_analisis, use_container_width=True, hide_index=True)

        # Gráfico de rendimientos
        df_graf = df_analisis[df_analisis["Rendimiento (%)"].str.contains("%", na=False)].copy()
        if not df_graf.empty:
            df_graf["Rend_num"] = df_graf["Rendimiento (%)"].str.rstrip('%').astype(float)
            fig_rend = go.Figure(go.Bar(
                x=df_graf["Periodo"],
                y=df_graf["Rend_num"],
                marker_color=['#26A65B' if x > 0 else '#C0392B' for x in df_graf["Rend_num"]],
                text=df_graf["Rend_num"].round(2),
                textposition='outside'
            ))
            fig_rend.update_layout(title="Rendimientos por Periodo",
                                   template="plotly_white", height=400, showlegend=False)
            fig_rend.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_rend, use_container_width=True)

        st.divider()

        # ==============================
        # INCOME STATEMENT
        # ==============================
        st.subheader("📘 Income Statement")
        df_income = obtener_income_yahoo(ticker)
        if not df_income.empty:
            st.dataframe(df_income, use_container_width=True, hide_index=True)

        st.divider()

        # ==============================
        # ANÁLISIS DE PEERS CON GEMINI
        # ==============================
        if peers and not df_comp.empty:
            st.subheader("🤝 Análisis Comparativo de Peers (Gemini AI)")
            
            with st.spinner("Obteniendo insights de peers y generando análisis..."):
                peers_insights = obtener_financial_insights_peers([ticker] + peers[:5])
                
                peers_data = f"""
                DATOS DE {ticker}:
                Market Cap: {info.get('marketCap')}, P/E: {info.get('trailingPE')}, ROE: {info.get('returnOnEquity')}, 
                Beta: {info.get('beta')}, EPS: {info.get('trailingEps')}, 
                Gross Margin: {info.get('grossMargins')}, Profit Margin: {info.get('profitMargins')}
                Financial Insight de {ticker}: {financial_insights if financial_insights else 'No disponible'}
                
                COMPARACIÓN CON PEERS:
                {df_comp.to_string()}
                
                FINANCIAL INSIGHTS DE PEERS:
                {chr(10).join([f"{t}: {insight}" for t, insight in peers_insights.items()])}
                
                RENDIMIENTOS HISTÓRICOS DE {ticker}:
                {df_analisis.to_string()}
                """
                
                modelo = genai.GenerativeModel("gemini-2.5-flash")
                prompt_peers = f"""
                Eres un analista financiero senior especializado en análisis comparativo de empresas.
                
                Analiza {ticker} y sus competidores principales con base en los siguientes datos:
                {peers_data}
                
                Proporciona un análisis comparativo estructurado en MÁXIMO 350 palabras que incluya:
                
                1. POSICIÓN COMPETITIVA
                2. FORTALEZAS RELATIVAS
                3. DEBILIDADES RELATIVAS
                4. OPORTUNIDADES Y RIESGOS
                5. RECOMENDACIÓN DE INVERSIÓN
                
                Toda la respuesta en texto plano, sin markdown.
                """
                
                respuesta_peers = modelo.generate_content(prompt_peers)
            
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 25px; border-radius: 12px; color: white; 
                            box-shadow: 0 6px 20px rgba(0,0,0,0.25);'>
                    <h3 style='margin-top: 0; color: white;'>🎯 Análisis Comparativo de Competidores</h3>
                    <div style='background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; 
                                backdrop-filter: blur(10px); line-height: 1.7;'>
                        {respuesta_peers.text.strip().replace(chr(10), '<br>')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # ==============================
        # ANÁLISIS PRINCIPAL CON GEMINI
        # ==============================
        st.subheader("🧠 Análisis Integral con Gemini AI")

        prompt = f"""
        Eres un analista financiero profesional. Analiza {ticker} con los siguientes datos:

        METRICAS CLAVE:
        Market Cap: {info.get('marketCap')}
        P/E: {info.get('trailingPE')}
        ROE: {info.get('returnOnEquity')}
        Beta: {info.get('beta')}
        EPS: {info.get('trailingEps')}
        Margen Bruto: {info.get('grossMargins')}
        Margen Neto: {info.get('profitMargins')}
        Deuda/Equity: {info.get('debtToEquity')}
        Sector: {info.get('sector')}
        Industria: {info.get('industry')}

        RENDIMIENTOS HISTÓRICOS:
        {df_analisis.to_string()}

        FINANCIAL INSIGHTS:
        {financial_insights if financial_insights else 'No disponible'}

        Proporciona un análisis de 280 palabras con:
        1. RECOMENDACIÓN
        2. FORTALEZAS
        3. RIESGOS
        4. VALORACIÓN
        5. PERSPECTIVA CORTO PLAZO
        6. PERSPECTIVA LARGO PLAZO
        
        Respuesta en texto plano.
        """

        with st.spinner("Generando análisis integral..."):
            respuesta = modelo.generate_content(prompt)

        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; border-radius: 12px; color: white; 
                        box-shadow: 0 6px 20px rgba(0,0,0,0.25);'>
                <h3 style='margin-top: 0; color: white;'>📊 Análisis Fundamental Completo</h3>
                <div style='background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; 
                            backdrop-filter: blur(10px); line-height: 1.7;'>
                    {respuesta.text.strip().replace(chr(10), '<br>')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning("⚠️ **Advertencia Legal:** Este análisis es generado por IA con fines informativos y educativos. NO constituye asesoramiento financiero profesional.")

    except Exception as e:
        st.error(f"❌ Error durante el análisis: {e}")
        st.info("💡 Verifica que el ticker sea correcto y esté disponible en Yahoo Finance")


# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:11px;'>
    📊 <b>Fuentes:</b> Yahoo Finance & Finviz | 🤖 <b>IA:</b> Gemini 2.5 Flash<br>
    🎓 <b>Proyecto:</b> Ingeniería Financiera | 💻 <b>Versión:</b> 3.3 | ⚖️ <b>Disclaimer:</b> Solo fines educativos
</div>
""", unsafe_allow_html=True)
