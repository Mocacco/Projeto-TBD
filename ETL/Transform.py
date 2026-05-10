import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ETL.Extract import extract_from_compressed
import re

def transform_com_diagnostico(df):
    print("\n" + "="*50)
    print("INICIANDO ETAPA DE TRANSFORM (LIMPEZA E QUALIDADE)")
    print("="*50)

    # 1. Diagnóstico Inicial (Passo "LP" do quadro)
    print(f"1. CONTAGEM DE REGISTROS: {df.shape[0]}")
    print(f"2. CONTAGEM DE COLUNAS: {df.shape[1]}")
    print(f"3. CHECAGEM DE TIPOS DE DADOS:\n{df.dtypes}")

    # 2. Valores Ausentes e Únicos (Coluna por Coluna)
    print("\n4. ANÁLISE DE VALORES POR COLUNA:")
    for col in df.columns:
        nulos = df[col].isnull().sum()
        unicos = df[col].nunique()
        print(f"   - Coluna '{col}': {nulos} nulos | {unicos} valores únicos")

    # 3. Retirada de Valores NAN (Exigência do quadro)
    if df.isnull().values.any():
        print("\n5. LIMPANDO VALORES NAN...")
        df = df.dropna(subset=['review'])
        print(f"   - Registros após remover NaNs: {len(df)}")
    else:
        print("\n5. NENHUM VALOR NAN DETECTADO.")

    # 4. Tradução/Mapeamento (Passo 3º do quadro)
    print("\n6. TRADUZINDO DATASET (Mapeamento de Humor)...")
    mapeamento = {'1': 'Negativo', '2': 'Positivo'}
    df['sentimento'] = df['label'].map(mapeamento)
    print(f"   - Labels mapeadas: {df['sentimento'].unique()}")

    # 5. Limpeza de Texto (NLP)
    print("\n7. PROCESSANDO TEXTO (NLP CLEANING)...")
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    df['review_clean'] = df['review'].apply(clean_text)

    # 6. Estatística Descritiva (Exigência do quadro)
    print("\n8. ESTATÍSTICA DESCRITIVA (Comprimento dos Reviews):")
    df['tamanho'] = df['review_clean'].str.len()
    print(df['tamanho'].describe())

    # 7. TF-IDF (Mineração de Gaps)
    print("\n9. GERANDO MATRIZ TF-IDF (EXTRAÇÃO DE ATRIBUTOS)...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(df['review_clean'])
    
    print("="*50)
    print("TRANSFORMAÇÃO CONCLUÍDA COM SUCESSO")
    print("="*50 + "\n")
    
    return matrix, df

if __name__ == "__main__":
    # 1. PASSO: EXTRAÇÃO (Extract)
    print("Iniciando processo...")
    dados_brutos = extract_from_compressed() 
    
    # 2. PASSO: CHAMAR A FUNÇÃO DE TRANSFORMAÇÃO 
    matriz, dados_finais = transform_com_diagnostico(dados_brutos)
    