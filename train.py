import pandas as pd
import glob
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# 1. Load data
def load_all_data(data_dir='data'):
    files = glob.glob(f"{data_dir}/*.xlsx")
    df_list = []
    for file in files:
        tmp = pd.read_excel(file)
        tmp['Date'] = pd.to_datetime(tmp['Date'])
        df_list.append(tmp)
    return pd.concat(df_list, ignore_index=True)

# 2. Build features (surface Elo, head-to-head, recent form)
def build_feature_dataset(df, initial_rating=1500, k_factor=32, recent_n=5):
    df = df.sort_values('Date')
    ratings = defaultdict(lambda: defaultdict(lambda: initial_rating))
    head2head = defaultdict(lambda: [0])  # wins count for (winner, loser)
    recent = defaultdict(lambda: deque(maxlen=recent_n))

    rows, labels = [], []
    for _, r in df.iterrows():
        w, l = r['Winner'], r['Loser']
        surface, series = r['Surface'], r['Series']

        Rw = ratings[surface][w]
        Rl = ratings[surface][l]
        Ew = 1 / (1 + 10 ** ((Rl - Rw) / 400))
        El = 1 - Ew

        # head-to-head diff
        h2h_diff = head2head[(w, l)][0] - head2head[(l, w)][0]

        # recent form diff
        rec_w = sum(recent[w]) / len(recent[w]) if recent[w] else 0.5
        rec_l = sum(recent[l]) / len(recent[l]) if recent[l] else 0.5
        rec_diff = rec_w - rec_l

        # Winner perspective
        rows.append({'Elo_diff': Rw - Rl,
                     'Head2Head_diff': h2h_diff,
                     'Recent_diff': rec_diff,
                     'Surface': surface,
                     'Series': series})
        labels.append(1)
        # Loser perspective
        rows.append({'Elo_diff': Rl - Rw,
                     'Head2Head_diff': -h2h_diff,
                     'Recent_diff': -rec_diff,
                     'Surface': surface,
                     'Series': series})
        labels.append(0)

        # Updates
        ratings[surface][w] += k_factor * (1 - Ew)
        ratings[surface][l] += k_factor * (0 - El)
        head2head[(w, l)][0] += 1
        recent[w].append(1)
        recent[l].append(0)

    return pd.DataFrame(rows), pd.Series(labels), ratings, head2head, recent

# Load and prepare
df_all = load_all_data()
X, y, ratings, head2head, recent = build_feature_dataset(df_all)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
cat_feats = ['Surface', 'Series']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
], remainder='passthrough')

# Base model pipeline
pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', GradientBoostingClassifier(random_state=42))
])

# Hyperparameter search space
param_dist = {
    'clf__n_estimators': [100, 200, 300],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__max_depth': [3, 5, 7]
}

search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist,
    n_iter=10, cv=3, scoring='accuracy',
    n_jobs=-1, random_state=42
)

# Run search
search.fit(X_train, y_train)
print(f"Best CV accuracy: {search.best_score_:.3f}")
print("Best params:", search.best_params_)

# Evaluate on test set
best_model = search.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f"Test set accuracy: {test_acc:.3f}")

# Convert structures to plain dicts for saving
ratings_dict = {s: dict(pdict) for s, pdict in ratings.items()}
head2head_dict = {k: v[0] for k, v in head2head.items()}
recent_dict = {p: list(dq) for p, dq in recent.items()}

# Save everything
joblib.dump(best_model, 'tennis_predictor_tuned.pkl')
joblib.dump(ratings_dict, 'elo_ratings_surface.pkl')
joblib.dump(head2head_dict, 'head2head.pkl')
joblib.dump(recent_dict, 'recent_form.pkl')

print("Saved tuned model and data.")