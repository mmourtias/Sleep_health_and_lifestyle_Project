import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Φόρτωση του καθαρισμένου dataset
df = pd.read_csv('data/Sleep_health_cleaned.csv')

# Επαλήθευση κατανομής (0: Insomnia, 1: None, 2: Sleep Apnea) [cite: 19]
print("Κατανομή Κλάσεων στο Dataset:")
print(df['Sleep Disorder_encoded'].value_counts())
print("-" * 30)

X = df.drop('Sleep Disorder_encoded', axis=1)
y = df['Sleep Disorder_encoded']

# Δημιουργία φακέλου output αν δεν υπάρχει [cite: 54]
if not os.path.exists('output'):
    os.makedirs('output')

# 2. Ορισμός των μοντέλων και των παραμέτρων για Tuning [cite: 24]
# Χρησιμοποιούμε Stratified 10-Fold CV για την αξιολόγηση 
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model_params = {
    "Decision Tree (C4.5)": {
        "model": DecisionTreeClassifier(criterion='entropy', random_state=42),
        "params": {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11, 13],
            "weights": ['uniform', 'distance']
        }
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {} # Δεν έχει σημαντικές παραμέτρους για tuning
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None]
        }
    },
    "ADABOOST": {
        "model": AdaBoostClassifier(random_state=42, algorithm='SAMME'),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0]
        }
    }
}

# 3. Εκτέλεση GridSearchCV για κάθε μοντέλο
best_models = {}
comparison_data = []

print("Έναρξη Hyperparameter Tuning με 10-Fold CV...")

for name, mp in model_params.items():
    grid = GridSearchCV(mp['model'], mp['params'], cv=skf, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    
    best_models[name] = grid.best_estimator_
    comparison_data.append({
        "Model": name,
        "Best Accuracy": grid.best_score_,
        "Best Params": grid.best_params_
    })
    print(f"Ολοκληρώθηκε: {name} | Best Score: {grid.best_score_:.4f}")

# 4. Συγκριτικός Πίνακας
results_df = pd.DataFrame(comparison_data).sort_values(by="Best Accuracy", ascending=False)
print("\n--- ΤΕΛΙΚΟΣ ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ (ΜΕ TUNING) ---")
print(results_df[["Model", "Best Accuracy", "Best Params"]].to_string(index=False))

# Αποθήκευση αποτελεσμάτων
results_df.to_csv('output/model_tuning_results.csv', index=False)

# 5. Οπτικοποίηση Σύγκρισης
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Model", y="Best Accuracy", palette="viridis")
plt.title("Σύγκριση Αλγορίθμων μετά το Hyperparameter Tuning")
plt.ylabel("Μέση Ακρίβεια (10-Fold CV)")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('output/classification_tuning_comparison.png')

print("\nΤο γράφημα και τα αποτελέσματα αποθηκεύτηκαν στον φάκελο 'output/'.")