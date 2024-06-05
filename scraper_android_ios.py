import pandas as pd
import numpy as np
from google_play_scraper import reviews, Sort
from app_store_scraper import AppStore
import matplotlib.pyplot as plt  # Certifique-se de que matplotlib está instalado

# Configurações do aplicativo
google_play_app_id = 'br.com.vivo.vivoeasy'  # ID correto do aplicativo no Google Play
app_store_app_id = 'id1049958200'  # ID correto do aplicativo na App Store (substitua pelo ID real)
lang = 'pt'  # Idioma português
country = 'br'  # País Brasil

# Função para extrair e salvar análises do Google Play
def scrape_google_play_reviews(app_id, lang, country, count=5):
    result, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=count
    )

    if not result:
        print("Nenhuma análise encontrada no Google Play.")
        return None
    else:
        reviews_df = pd.DataFrame(np.array(result), columns=['review'])
        reviews_df_expanded = reviews_df.join(pd.DataFrame(reviews_df.pop('review').tolist()))
        reviews_df_expanded.to_csv('google_play_reviews.csv', index=False)
        print("Análises do Google Play salvas em 'google_play_reviews.csv'")
        return reviews_df_expanded

# Função para extrair e salvar análises da App Store
def scrape_app_store_reviews(app_id, country, count=5):
    app = AppStore(country=country, app_name=app_id)
    app.review(how_many=count)

    if not app.reviews:
        print("Nenhuma análise encontrada na App Store.")
        return None
    else:
        reviews_df = pd.DataFrame(app.reviews)
        reviews_df.to_csv('app_store_reviews.csv', index=False)
        print("Análises da App Store salvas em 'app_store_reviews.csv'")
        return reviews_df

# Extrai e salva análises do Google Play
google_play_reviews_df = scrape_google_play_reviews(google_play_app_id, lang, country)

# Extrai e salva análises da App Store
app_store_reviews_df = scrape_app_store_reviews(app_store_app_id, country)

# Exibe as primeiras linhas dos DataFrames resultantes
if google_play_reviews_df is not None:
    print("Google Play Reviews:")
    print(google_play_reviews_df.head())

if app_store_reviews_df is not None:
    print("App Store Reviews:")
    print(app_store_reviews_df.head())