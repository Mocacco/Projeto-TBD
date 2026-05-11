import pandas as pd

df = pd.read_parquet('amazon_reviews.parquet')

print(df.head(50))