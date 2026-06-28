import pandas as pd
import numpy as np

def analyze_score_distributions():
    df = pd.read_csv('cleaned_eju_data.csv')
    
    # Calculate representative scores for each row
    # Since they can take EJU 1 or EJU 2, let's find the max score they had for each subject
    df['日本語'] = np.maximum(df['日本語_1'].fillna(0), df['日本語_2'].fillna(0))
    df['記述'] = np.maximum(df['記述_1'].fillna(0), df['記述_2'].fillna(0))
    df['物理'] = np.maximum(df['物理_1'].fillna(0), df['物理_2'].fillna(0))
    df['化学'] = np.maximum(df['化学_1'].fillna(0), df['化学_2'].fillna(0))
    df['生物'] = np.maximum(df['生物_1'].fillna(0), df['生物_2'].fillna(0))
    df['综合'] = np.maximum(df['総合_1'].fillna(0), df['総合_2'].fillna(0))
    df['数１'] = np.maximum(df['数１_1'].fillna(0), df['数１_2'].fillna(0))
    df['数２'] = np.maximum(df['数２_1'].fillna(0), df['数２_2'].fillna(0))
    
    # Classify track:
    # If Sogou > 0 -> Liberal Arts (文科)
    # If any Science subject > 0 -> Science (理科)
    # If both are 0, look at math
    df['is_bunka'] = df['综合'] > 0
    df['is_rika'] = (df['物理'] > 0) | (df['化学'] > 0) | (df['生物'] > 0)
    
    # Total EJU scores
    df['文科EJU总分'] = df['日本語'] + df['综合'] + df['数１']
    # For rika, sum of top 2 science subjects + math2 + ja
    science_sum = df[['物理', '化学', '生物']].fillna(0).values
    science_sum.sort(axis=1) # sort ascending, so last two are the top 2
    df['理科EJU总分'] = df['日本語'] + science_sum[:, -1] + science_sum[:, -2] + df['数２']
    
    print("\n--- Liberal Arts (文科) Score Analysis ---")
    bunka_df = df[df['is_bunka'] & (df['文科EJU总分'] > 300)]
    print(f"Total Liberal Arts records: {len(bunka_df)}")
    
    for school in ['東京大学', '早稲田大学', '明治大学', '日本大学']:
        school_df = bunka_df[bunka_df['受験校'].str.contains(school, na=False)]
        if len(school_df) > 0:
            passed = school_df[school_df['合否'] == 1]
            failed = school_df[school_df['合否'] == 0]
            print(f"\nUniversity: {school}")
            print(f"Passed count: {len(passed)}, Failed count: {len(failed)}")
            if len(passed) > 0:
                print(f"Passed average EJU total: {passed['文科EJU总分'].mean():.1f} (min: {passed['文科EJU总分'].min()}, max: {passed['文科EJU总分'].max()})")
            if len(failed) > 0:
                print(f"Failed average EJU total: {failed['文科EJU总分'].mean():.1f} (min: {failed['文科EJU总分'].min()}, max: {failed['文科EJU总分'].max()})")

    print("\n--- Science (理科) Score Analysis ---")
    rika_df = df[df['is_rika'] & (df['理科EJU总分'] > 300)]
    print(f"Total Science records: {len(rika_df)}")
    
    for school in ['東京大学', '早稲田大学', '東京理科大学', '日本大学']:
        school_df = rika_df[rika_df['受験校'].str.contains(school, na=False)]
        if len(school_df) > 0:
            passed = school_df[school_df['合否'] == 1]
            failed = school_df[school_df['合否'] == 0]
            print(f"\nUniversity: {school}")
            print(f"Passed count: {len(passed)}, Failed count: {len(failed)}")
            if len(passed) > 0:
                print(f"Passed average EJU total: {passed['理科EJU总分'].mean():.1f} (min: {passed['理科EJU总分'].min()}, max: {passed['理科EJU总分'].max()})")
            if len(failed) > 0:
                print(f"Failed average EJU total: {failed['理科EJU总分'].mean():.1f} (min: {failed['理科EJU总分'].min()}, max: {failed['理科EJU总分'].max()})")

if __name__ == '__main__':
    analyze_score_distributions()
