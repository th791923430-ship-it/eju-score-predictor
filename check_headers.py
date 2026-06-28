import pandas as pd

files = [
    '2017年度　そうがくしゃデータ印刷用(1).xlsx',
    '2018そうがくしゃデータ(1).xlsx',
    '2020そうがくしゃデータ(1).xlsx',
    '2021年学部そうがくしゃ データ(2).xlsx'
]

headers = [1, 0, 0, 0]

for f, h in zip(files, headers):
    try:
        df = pd.read_excel(f, nrows=0, header=h)
        print(f"\n--- {f} ---")
        print(df.columns.tolist())
    except Exception as e:
        print(f"Failed to read {f}: {e}")
