import pandas as pd
import numpy as np

def analyze_data():
    df = pd.read_csv('cleaned_eju_data.csv')
    print("Columns loaded:", df.columns.tolist())
    
    # Check null values in major columns
    print("\nNull counts in key fields:")
    print(df[['受験校', '学部／研究科', '合否']].isnull().sum())
    
    # Fill nan in strings
    df['受験校'] = df['受験校'].fillna('未知大学')
    df['学部／研究科'] = df['学部／研究科'].fillna('未知学部')
    df['学科／攻'] = df['学科／専攻'].fillna('未知学科')
    
    # Check top universities
    print("\nTop 15 Universities by record count:")
    print(df['受験校'].value_counts().head(15))
    
    # Check score completeness
    print("\nCompleteness of EJU 1 (June) vs EJU 2 (Nov):")
    has_eju1 = df['日本語_1'].notnull() | df['総合_1'].notnull() | df['数１_1'].notnull() | df['数２_1'].notnull()
    has_eju2 = df['日本語_2'].notnull() | df['総合_2'].notnull() | df['数１_2'].notnull() | df['数２_2'].notnull()
    
    print(f"Has EJU 1 only: {sum(has_eju1 & ~has_eju2)}")
    print(f"Has EJU 2 only: {sum(~has_eju1 & has_eju2)}")
    print(f"Has Both: {sum(has_eju1 & has_eju2)}")
    print(f"Has Neither (only JLPT/English?): {sum(~has_eju1 & ~has_eju2)}")
    
    # Let's inspect some samples of popular schools like Tokyo University (東京大学) or Waseda (早稲田大学) if present
    for school in ['東京大学', '京都大学', '早稲田大学', '慶應義塾大学', '明治大学', '日本大学']:
        school_df = df[df['受験校'].str.contains(school, na=False)]
        if len(school_df) > 0:
            print(f"\n--- {school} ---")
            print(f"Total applicants: {len(school_df)}")
            print(f"Passed: {sum(school_df['合否'] == 1)}, Failed: {sum(school_df['合否'] == 0)}")
            # Show average score (taking max of June or Nov for Japanese and Sogou as approximation)
            ja_scores = np.maximum(school_df['日本語_1'].fillna(0), school_df['日本語_2'].fillna(0))
            ja_scores = ja_scores[ja_scores > 0]
            if len(ja_scores) > 0:
                print(f"Average Japanese score (max of June/Nov): {ja_scores.mean():.1f}")
            
if __name__ == '__main__':
    analyze_data()
