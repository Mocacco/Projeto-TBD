import pandas as pd
import zipfile
import bz2

def extract_from_compressed():
    
    with zipfile.ZipFile('test.ft.txt.bz2.zip', 'r') as z:
        nome_bz2 = z.namelist()[0]
        with z.open(nome_bz2) as f_zip:
            # 2. Lemos o conteúdo bz2 diretamente para o Pandas
            df_raw = pd.read_csv(
                f_zip, 
                header=None, 
                sep='\t', 
                names=['raw_text'], 
                compression='bz2',
                engine='python',
                quoting=3,
                nrows=100000 
            )
    
    # 3. Parsing das colunas 
    df_extracted = df_raw['raw_text'].str.split(' ', n=1, expand=True)
    df_extracted.columns = ['label', 'review']
    
    # Limpeza básica das labels
    df_extracted['label'] = df_extracted['label'].str.replace('__label__', '')
    
    return df_extracted

# Execução do teste de extração
if __name__ == "__main__":
    dados = extract_from_compressed()
    print("Colunas extraídas:", dados.columns.tolist())
    print("Exemplo de dado:\n", dados.head())
    dados['label'] = dados['label'].str.strip()
    dados['review'] = dados['review'].str.strip()
    # Salvar em Parquet 
    dados.to_parquet('dados_extraidos.parquet')
    print("\nArquivo salvo em .parquet para otimização de armazenamento.")