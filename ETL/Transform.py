import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def transform_com_diagnostico(df):

    print("\n" + "=" * 50)
    print("INICIANDO ETAPA DE TRANSFORM (LIMPEZA E QUALIDADE)")
    print("=" * 50)

    # 1. Diagnóstico e Limpeza
    print(f"1. REGISTROS INICIAIS: {len(df)}")

    # Valores Ausentes e Únicos
    for col in df.columns:
        nulos = df[col].isnull().sum()

        print(
            f"   - Coluna '{col}': "
            f"{nulos} nulos | "
            f"{df[col].nunique()} únicos"
        )

    # Retirada de NaNs e duplicados
    df = df.dropna(subset=['review']).drop_duplicates().copy()

    print(
        f"2. REGISTROS APÓS LIMPEZA "
        f"(NaN/Duplicados): {len(df)}"
    )

    # 2. Tradução / Mapeamento
    mapeamento = {
        '1': 'Negativo',
        '2': 'Positivo'
    }

    df['sentimento'] = df['label'].map(mapeamento)

    # 3. Processamento NLP
    print("3. PROCESSANDO TEXTO (NLP CLEANING)...")

    def clean_text(text):

        text = str(text).lower()

        # Remove pontuação
        text = re.sub(r'[^\w\s]', '', text)

        # Remove espaços duplicados
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # SOBRESCREVE A COLUNA ORIGINAL
    df['review'] = df['review'].apply(clean_text)

    # Feature de tamanho
    df['tamanho'] = df['review'].str.len()

    # 4. Estatística Descritiva
    print("\n4. ESTATÍSTICA DESCRITIVA (Comprimento):")

    print(df['tamanho'].describe())

    # 5. Mineração de Gaps (TF-IDF)
    print("\n5. EXTRAINDO TERMOS COM TF-IDF...")

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )

    matrix = tfidf.fit_transform(df['review'])

    # Reviews negativos
    df_neg = df[df['label'] == '1']

    if not df_neg.empty:

        tfidf_neg = TfidfVectorizer(
            stop_words='english',
            max_features=10
        )

        tfidf_neg.fit(df_neg['review'])

        print(
            "   - Termos indicadores de Gaps:"
        )

        print(
            list(tfidf_neg.get_feature_names_out())
        )

    print("=" * 50)

    return matrix, df