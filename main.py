from ETL.Extract import extract_from_compressed
from ETL.Transform import transform_com_diagnostico
from ETL.Load import executar_load

if __name__ == "__main__":
    print("Iniciando Pipeline ETL\n")

    # 1. Extração
    dados_brutos = extract_from_compressed()

    # 2. Transformação (Aqui está o segredo!)
    matriz_tfidf, dados_finalizados = transform_com_diagnostico(dados_brutos)

    # 3. Carga
    executar_load(dados_finalizados)

    print("\nPipeline finalizado com sucesso!")