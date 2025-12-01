import yfinance as yf

def obtener_noticias_yf(ticker):
    """Obtiene noticias recientes de Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        news = t.news
        
        if not news:
            return []
        
        # Procesar las noticias al nuevo formato
        noticias_procesadas = []
        for item in news:
            try:
                # El nuevo formato tiene estructura anidada
                if isinstance(item, dict):
                    # Extraer contenido principal
                    content = item.get('content', {})
                    
                    noticia = {
                        'title': content.get('title', 'Sin título'),
                        'link': content.get('canonicalUrl', {}).get('url', content.get('clickThroughUrl', {}).get('url', '#')),
                        'publisher': content.get('provider', {}).get('displayName', 'Desconocido'),
                        'published': content.get('pubDate', ''),
                        'summary': content.get('summary', '')
                    }
                    noticias_procesadas.append(noticia)
            except Exception as e:
                # Si hay error procesando una noticia específica, continuar con las demás
                continue
        
        return noticias_procesadas if noticias_procesadas else []
    except Exception as e:
        return []

# Test
noticias = obtener_noticias_yf("AAPL")
print(f"\n✅ Total noticias obtenidas: {len(noticias)}\n")

if noticias:
    print("📰 Primeras 3 noticias:\n")
    for i, n in enumerate(noticias[:3], 1):
        print(f"{i}. {n['title']}")
        print(f"   🏢 Publisher: {n['publisher']}")
        print(f"   🔗 Link: {n['link']}")
        print(f"   📝 Summary: {n['summary'][:80]}...")
        print()
else:
    print("❌ No se encontraron noticias")
