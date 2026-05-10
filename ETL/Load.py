import pandas as pd

def executar_load(df):
    """
    ETAPA DE LOAD (Carga)
    Salva os dados conforme a estrutura de arquivos do projeto.
    """
    print("\n" + "="*50)
    print("INICIANDO ETAPA DE LOAD")
    print("="*50)

    try:
        # 1. Salvando em Parquet (Otimizado para o pipeline)
        df.to_parquet('amazon_reviews.parquet', index=False)
        print("1. [SUCESSO] Arquivo 'amazon_reviews.parquet' gerado.")

        # 2. Salvando em CSV (Para visualização rápida)
        df.to_csv('amazon_reviews.csv', index=False, encoding='utf-8')
        print("2. [SUCESSO] Arquivo 'amazon_reviews.csv' gerado.")

        # 3. Salvando Versão Transformada (Se aplicável)
        df.to_parquet('dados_transformados.parquet', index=False)
        print("3. [SUCESSO] Arquivo 'dados_transformados.parquet' gerado.")

    except Exception as e:
        print(f"[ERRO] Falha ao salvar arquivos: {e}")

    print("="*50)

if __name__ == "_main_":
    print("Módulo Load pronto para uso.")
