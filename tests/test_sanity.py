def test_fixtures_sanity(sample_game_data, mock_schedule_response, temp_db_path):
    assert sample_game_data["game_pk"] == 748123
    assert sample_game_data["away_team"]["starter"]["name"] == "Gerrit Cole"
    assert sample_game_data["home_team"]["starter"]["name"] == "Brayan Bello"
    assert "Fenway Park" in sample_game_data["venue"]["name"]
    assert len(mock_schedule_response["dates"]) == 1
    assert temp_db_path.endswith(".db")
