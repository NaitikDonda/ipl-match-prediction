import pandas as pd
import joblib

# load models
model = joblib.load("ipl_model/model.pkl")
preprocessor = joblib.load("ipl_model/preprocessor.pkl")
match_df = pd.read_pickle("ipl_model/history.pkl")


def predict_ipl_winner(team1, team2, toss_winner, toss_decision, venue, city):
    # latest season
    season = match_df["season"].max()

    # basic features
    team1_is_toss = int(toss_winner == team1)
    toss_bat = int(toss_decision == "bat")

    # head-to-head
    h2h = match_df[
        ((match_df["team1"] == team1) & (match_df["team2"] == team2)) |
        ((match_df["team1"] == team2) & (match_df["team2"] == team1))
    ]

    h2h_wr = (h2h["match_won_by"] == team1).mean() if len(h2h) > 0 else 0.5

    # input dataframe
    data = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "city": city,
        "toss_decision": toss_decision,
        "team1_is_toss_winner": team1_is_toss,
        "toss_decision_bat": toss_bat,
        "h2h_win_rate_team1": h2h_wr,
        "h2h_total_matches": len(h2h),
        "form_team1_last5": 0.5,
        "form_team2_last5": 0.5,
        "form_diff": 0,
        "venue_wr_team1": 0.5,
        "venue_wr_team2": 0.5,
        "match_month": 4,
        "match_day_of_week": 5,
        "season": season
    }])

    # prediction
    X = preprocessor.transform(data)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    return {
        "winner": team1 if pred == 1 else team2,
        "team1_prob": round(prob[1] * 100, 2),
        "team2_prob": round(prob[0] * 100, 2)
    }