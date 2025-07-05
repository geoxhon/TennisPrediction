import joblib
import pandas as pd

# Load trained model and auxiliary data
model = joblib.load('tennis_predictor_tuned.pkl')
elo_ratings = joblib.load('elo_ratings_surface.pkl')
head2head = joblib.load('head2head.pkl')
recent_form = joblib.load('recent_form.pkl')

def predict_match(player1, player2, surface, series):
    """
    Predicts the outcome of a match between player1 and player2.

    Args:
        player1 (str): Name of the first player.
        player2 (str): Name of the second player.
        surface (str): Surface type (e.g., 'Clay', 'Hard', 'Grass').
        series (str): Tournament series (e.g., 'Grand Slam', 'ATP 500').

    Returns:
        winner (str): Predicted winner's name.
        probability (float): Probability of player1 winning.
    """
    # Surface-specific Elo difference
    r1 = elo_ratings.get(surface, {}).get(player1, 1500)
    r2 = elo_ratings.get(surface, {}).get(player2, 1500)
    elo_diff = r1 - r2

    # Head-to-head difference
    hh_diff = head2head.get((player1, player2), 0) - head2head.get((player2, player1), 0)

    # Recent form difference
    rec1 = recent_form.get(player1, [])
    rec2 = recent_form.get(player2, [])
    rec_r1 = sum(rec1) / len(rec1) if rec1 else 0.5
    rec_r2 = sum(rec2) / len(rec2) if rec2 else 0.5
    recent_diff = rec_r1 - rec_r2

    # Build DataFrame for model input
    df = pd.DataFrame([{  
        'Elo_diff': elo_diff,
        'Head2Head_diff': hh_diff,
        'Recent_diff': recent_diff,
        'Surface': surface,
        'Series': series
    }])

    # Predict probability that player1 wins
    prob = model.predict_proba(df)[0][1]
    winner = player1 if prob > 0.5 else player2
    prob = prob if prob > 0.5 else 1 - prob
    return winner, prob

# Example usage
if __name__ == '__main__':
    p1, p2 = 'Dimitrov G.', 'Ofner S.'
    surf, ser = 'Grass', 'Grand Slam'
    win, pr = predict_match(p1, p2, surf, ser)
    print(f"Predicted winner: {win} with probability {pr:.2f}")