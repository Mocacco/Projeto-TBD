import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def transform_data(df):
    """
    Etapa de Transformação: Limpeza e Vetorização TF-IDF
    """
    print("Iniciando a transformação dos dados...")

    # 1. Limpeza de Texto básica
    def clean_text(text):
        text = text.lower() # Padroniza para minúsculas
        text = re.sub(r'[^\w\s]', '', text) # Remove pontuação
        text = re.sub(r'\d+', '', text) # Remove números
        return text

    df['review_clean'] = df['review'].apply(clean_text)

    # 2. Configuração do TF-IDF
    # stop_words='english' remove palavras comuns (the, and, is)
    # max_features=1000 limita às 1000 palavras mais importantes para performance
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))

    # 3. Gerando a Matriz TF-IDF
    tfidf_matrix = tfidf.fit_transform(df['review_clean'])
    
    # Criando um DataFrame com os pesos das palavras para visualização
    weights = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=tfidf.get_feature_names_out()
    )

    print("Transformação concluída com sucesso!")
    return weights, tfidf