import pandas as pd
import numpy as np

def test_prediction_engine():
    df = pd.read_csv('cleaned_eju_data.csv')
    
    # Fill NaN and compute max scores
    df['日本語'] = np.maximum(df['日本語_1'].fillna(0), df['日本語_2'].fillna(0))
    df['综合'] = np.maximum(df['総合_1'].fillna(0), df['総合_2'].fillna(0))
    df['数１'] = np.maximum(df['数１_1'].fillna(0), df['数１_2'].fillna(0))
    df['数２'] = np.maximum(df['数２_1'].fillna(0), df['数２_2'].fillna(0))
    df['物理'] = np.maximum(df['物理_1'].fillna(0), df['物理_2'].fillna(0))
    df['化学'] = np.maximum(df['化学_1'].fillna(0), df['化学_2'].fillna(0))
    df['生物'] = np.maximum(df['生物_1'].fillna(0), df['生物_2'].fillna(0))
    
    df['is_bunka'] = df['综合'] > 0
    df['is_rika'] = (df['物理'] > 0) | (df['化学'] > 0) | (df['生物'] > 0)
    
    df['文科EJU总分'] = df['日本語'] + df['综合'] + df['数１']
    science_sum = df[['物理', '化学', '生物']].fillna(0).values
    science_sum.sort(axis=1)
    df['理科EJU总分'] = df['日本語'] + science_sum[:, -1] + science_sum[:, -2] + df['数２']
    
    # Let's write a predictor function
    def predict_probability(user_score, school_name, faculty_name, is_bunka=True):
        # Filter historical records for this school + faculty
        school_mask = df['受験校'].str.contains(school_name, na=False)
        faculty_mask = df['学部／研究科'].str.contains(faculty_name, na=False) if pd.notnull(faculty_name) else True
        track_mask = df['is_bunka'] if is_bunka else df['is_rika']
        
        hist = df[school_mask & faculty_mask & track_mask]
        
        # Fallback if too few records
        if len(hist) < 8:
            hist = df[school_mask & track_mask] # fallback to school level
        if len(hist) < 5:
            # global fallback based on school tier
            return 0.5, "数据较少，无法精确评估"
            
        score_col = '文科EJU总分' if is_bunka else '理科EJU总分'
        
        # Extract passed scores
        passed_scores = hist[hist['合否'] == 1][score_col].dropna()
        failed_scores = hist[hist['合否'] == 0][score_col].dropna()
        
        if len(passed_scores) < 3:
            # Fallback to general stats
            return 0.5, "合格数据不足，无法评估"
            
        median_passed = passed_scores.median()
        std_passed = passed_scores.std()
        if pd.isna(std_passed) or std_passed == 0:
            std_passed = 20.0 # default std deviation
            
        z = (user_score - median_passed) / std_passed
        # Sigmoid function
        prob = 1.0 / (1.0 + np.exp(-(1.5 * z + 0.5)))
        
        # Limit to 5% to 95% to avoid absolute certainty
        prob = np.clip(prob, 0.05, 0.95)
        
        return prob, f"基于 {len(hist)} 条历史记录评估 (合格中位数: {median_passed:.1f}, 标准差: {std_passed:.1f})"

    # Test for some scores
    test_cases = [
        (720, '東京大学', '文', True),
        (750, '東京大学', '文', True),
        (680, '東京大学', '文', True),
        (680, '早稲田大学', '政治経済', True),
        (640, '早稲田大学', '政治経済', True),
        (640, '明治大学', '政治経済', True),
        (580, '明治大学', '政治経済', True),
        (550, '日本大学', '経済', True),
    ]
    
    print("\n--- Testing Prediction Formula ---")
    for score, school, faculty, is_bunka in test_cases:
        prob, note = predict_probability(score, school, faculty, is_bunka)
        print(f"User Score: {score} | {school} - {faculty} | Prob: {prob*100:.1f}% | {note}")

if __name__ == '__main__':
    test_prediction_engine()
