import app as app_module


def test_target_setting_mode_returns_required_score_ranges():
    app_module.load_and_precompute()
    app_module.app.testing = True
    client = app_module.app.test_client()

    response = client.post(
        '/api/predict',
        json={
            'mode': 'target_setting',
            'track': 'bunka',
            'target_school': '東京大学',
            'target_faculty': '文科一類-法'
        }
    )

    assert response.status_code == 200
    data = response.get_json()
    assert 'target_result' in data

    target = data['target_result']
    assert not target.get('error'), target
    assert target['school'] == '東京大学'
    assert 'stats' in target
    assert 'subject_scores' in target['stats']
    assert len(target['stats']['subject_scores']) > 0

    first_subject = target['stats']['subject_scores'][0]
    assert 'name' in first_subject
    assert 'p10' in first_subject and 'p90' in first_subject
