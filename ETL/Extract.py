import pandas as pd
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_from_compressed():
    # Extração robusta de arquivo compactado bz2 dentro de zip
    with zipfile.ZipFile('test.ft.txt.bz2.zip', 'r') as z:
        nome_bz2 = z.namelist()[0]
        with z.open(nome_bz2) as f_zip:
            df_raw = pd.read_csv(
                f_zip, header=None, sep='\t', names=['raw_text'], 
                compression='bz2', engine='python', quoting=3, nrows=100000 
            )
    
    # Parsing do formato fastText para Colunas
    df_extracted = df_raw['raw_text'].str.split(' ', n=1, expand=True)
    df_extracted.columns = ['label', 'review']
    df_extracted['label'] = df_extracted['label'].str.replace('__label__', '').str.strip()
    df_extracted['review'] = df_extracted['review'].str.strip()
    return df_extracted

def realizar_eda_completa(df):
    print("--- 1. DIAGNÓSTICO DO DATASET (Coluna por Coluna) ---")
    # Contar registros, colunas, tipos, ausentes e únicos
    diagnostico = pd.DataFrame({
        'Tipo': df.dtypes,
        'Valores Ausentes': df.isnull().sum(),
        'Valores Únicos': df.nunique()
    })
    print(f"Total Registros: {len(df)} | Total Colunas: {len(df.columns)}")
    print(diagnostico)

    print("\n--- 2. LIMPEZA (Retirada de NaNs e Duplicados) ---")
    df_limpo = df.dropna().drop_duplicates().copy()
    print(f"Registros após limpeza: {len(df_limpo)}")

    print("\n--- 3. TRADUÇÃO E ESTATÍSTICA DESCRIÇÃO ---")
    # Tradução de Labels
    mapeamento = {'1': 'Negativo', '2': 'Positivo'}
    df_limpo['sentimento_label'] = df_limpo['label'].map(mapeamento)
    
    # Estatística de texto: Tamanho dos reviews
    df_limpo['char_count'] = df_limpo['review'].str.len()
    print(df_limpo[['char_count']].describe())
    
    return df_limpo

def transform_tfidf(df):
    print("\n--- 4. TRANSFORM (TF-IDF para Gaps) ---")
    # Foco apenas em reviews negativos para encontrar Gaps de Produto
    df_neg = df[df['label'] == '1'].copy()
    tfidf = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf.fit(df_neg['review'])
    top_words = tfidf.get_feature_names_out()
    print(f"Palavras indicadoras de Gaps: {list(top_words)}")
    return top_words

if __name__ == "__main__":
    # Passo 1: Extrair
    dados_brutos = extract_from_compressed()
    
    # Passo 2: Limpar e Diagnosticar (Roteiro do Quadro)
    dados_processados = realizar_eda_completa(dados_brutos)
    
    # Passo 3: Transformar (Mineração)
    transform_tfidf(dados_processados)
    
    # Passo 4: Load (Carga Final)
    dados_processados.to_parquet('amazon_reviews.parquet')
    dados_processados.to_csv('amazon_reviews.csv', index=False)
    print("\nProcesso concluído: Diagnóstico realizado e arquivos salvos.")