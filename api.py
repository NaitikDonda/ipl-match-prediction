from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Get absolute path for Render compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Load models with proper paths
try:
    model = joblib.load(os.path.join(BASE_DIR, "ipl_model", "model.pkl"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "ipl_model", "preprocessor.pkl"))
    match_df = pd.read_pickle(os.path.join(BASE_DIR, "ipl_model", "history.pkl"))
    live_model = joblib.load(os.path.join(BASE_DIR, "ipl_model", "live_model.pkl"))
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    preprocessor = None
    match_df = None
    live_model = None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    team1 = data["team1"]
    team2 = data["team2"]
    toss_winner = data["toss_winner"]
    toss_decision = data["toss_decision"]
    venue = data["venue"]
    city = data["city"]

    team1_is_toss = int(toss_winner == team1)

    input_df = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "city": city,
        "toss_decision": toss_decision,
        "team1_is_toss_winner": team1_is_toss,
        "toss_decision_bat": int(toss_decision == "bat"),
        "h2h_win_rate_team1": 0.5,
        "h2h_total_matches": 0,
        "form_team1_last5": 0.5,
        "form_team2_last5": 0.5,
        "form_diff": 0,
        "venue_wr_team1": 0.5,
        "venue_wr_team2": 0.5,
        "match_month": 4,
        "match_day_of_week": 5,
        "season": match_df["season"].max()
    }])

    X = preprocessor.transform(input_df)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    return jsonify({
        "winner": team1 if pred == 1 else team2,
        "team1_prob": float(prob[1]*100),
        "team2_prob": float(prob[0]*100)
    })

@app.route("/predict_live", methods=["POST"])
def predict_live():
    data = request.json

    batting_team = data["batting_team"]
    bowling_team = data["bowling_team"]
    runs_scored = data["runs_scored"]
    balls_bowled = data["balls_bowled"]
    wickets_fallen = data["wickets_fallen"]
    runs_target = data["runs_target"]

    runs_left = max(runs_target - runs_scored, 0)
    balls_left = max(120 - balls_bowled, 0)
    overs_played = balls_bowled / 6
    overs_left = balls_left / 6

    crr = runs_scored / overs_played if overs_played > 0 else 0
    rrr = runs_left / overs_left if overs_left > 0 else (999 if runs_left > 0 else 0)

    # Use actual ML model for live prediction
    if live_model:
        input_df = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "runs_scored": runs_scored,
            "balls_bowled": balls_bowled,
            "wickets_fallen": wickets_fallen,
            "runs_target": runs_target,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "overs_played": overs_played,
            "overs_left": overs_left,
            "crr": crr,
            "rrr": rrr,
            "wickets_remaining": 10 - wickets_fallen
        }])
        
        # Get prediction from live model
        chasing_prob = float(live_model.predict(input_df)[0])
    else:
        # Fallback to simple calculation if model fails
        if runs_left <= 0:
            chasing_prob = 99.0
        elif balls_left <= 0:
            chasing_prob = 1.0
        else:
            ratio = rrr / (crr if crr > 0 else 1)
            chasing_prob = min(95, max(5, round(100 / (1 + pow(ratio, 1.4)))))

    situation = (
        "Chasing team wins!" if runs_left <= 0 else
        "Defending team wins!" if balls_left <= 0 else
        f"Need {runs_left} runs in {balls_left} balls"
    )

    return jsonify({
        "chasing_team_win_prob": chasing_prob,
        "defending_team_win_prob": 100 - chasing_prob,
        "status_snapshot": situation,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "crr": round(crr, 2),
        "rrr": round(rrr, 2) if rrr <= 50 else 999
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)