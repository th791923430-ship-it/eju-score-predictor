import pandas as pd
import numpy as np

def inspect_file(filename, header_row, pass_col='合否'):
    print(f"\n--- Inspecting {filename} ---")
    df = pd.read_excel(filename, header=header_row)
    print("Columns:", df.columns.tolist())
    print("Unique values in Pass/Fail column:", df[pass_col].unique() if pass_col in df.columns else "Not Found")
    print("Shape:", df.shape)
    
inspect_file('2023進学調査（そうがく社）.xlsx', 1)
inspect_file('2024進学調査（そうがく社）.xlsx', 1)
inspect_file('2025進学調査（そうがく社）.xlsx', 3) # Let's see: row index 2 was the header. When header=3 in pandas, it is index 3 or index 2? Let's check.
