import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Φόρτωση του καθαρισμένου dataset
if not os.path.exists('data/Sleep_health_cleaned.csv'):
    print("Σφάλμα: Δεν βρέθηκε το αρχείο 'data/Sleep_health_cleaned.csv'. Τρέξτε πρώτα το 01_preprocessing.py")
    exit()

df = pd.read_csv('data/Sleep_health_cleaned.csv')

# Επαλήθευση κατανομής κλάσεων
print("Κατανομή Κλάσεων (0: Insomnia, 1: None, 2: Sleep Apnea):")
print(df['Sleep Disorder_encoded'].value_counts())
print("-" * 30)

X = df.drop('Sleep Disorder_encoded', axis=1)
y = df['Sleep Disorder_encoded']

# Δημιουργία φακέλου output αν δεν υπάρχει
if not os.path.exists('output'):
    os.makedirs('output')

# 2. Ορισμός των μοντέλων και των παραμέτρων για Tuning
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model_params = {
    "Decision Tree (C4.5)": {
        "model": DecisionTreeClassifier(criterion='entropy', random_state=42),
        "params": {
            "max_depth": [3, 5, 8, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "KNN": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ]),
        "params": {
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ['uniform', 'distance']
        }
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {} # Default parameters
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None]
        }
    },
    "ADABOOST": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0]
        }
    }
}

# 3. Εκτέλεση GridSearchCV για κάθε μοντέλο (Multi-Metric)
best_models = {}
comparison_data = []
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

print("Έναρξη Hyperparameter Tuning με 10-Fold CV...")

for name, mp in model_params.items():
    grid = GridSearchCV(mp['model'], mp['params'], cv=skf, scoring=scoring_metrics, refit='accuracy', n_jobs=-1)
    grid.fit(X, y)
    
    best_index = grid.best_index_
    best_models[name] = grid.best_estimator_
    
    comparison_data.append({
        "Model": name,
        "Accuracy": grid.cv_results_['mean_test_accuracy'][best_index],
        "Precision (Macro)": grid.cv_results_['mean_test_precision_macro'][best_index],
        "Recall (Macro)": grid.cv_results_['mean_test_recall_macro'][best_index],
        "F1-Score (Macro)": grid.cv_results_['mean_test_f1_macro'][best_index],
        "Best Params": grid.best_params_
    })
    print(f"Ολοκληρώθηκε: {name}")

# 4. Τελικός Συγκριτικός Πίνακας
results_df = pd.DataFrame(comparison_data).sort_values(by="Accuracy", ascending=False)
print("\n--- ΤΕΛΙΚΟΣ ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ (ΑΠΟ CROSS-VALIDATION) ---")
print(results_df[["Model", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]].to_string(index=False))

# 5. Οπτικοποίηση Σύγκρισης
plt.figure(figsize=(12, 6))
plot_df = results_df.melt(id_vars="Model", value_vars=["Accuracy", "F1-Score (Macro)"], var_name="Metric", value_name="Score")
sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
plt.title("Σύγκριση Αλγορίθμων (Accuracy vs F1-Score)")
plt.ylim(0.7, 1.0)
plt.savefig('output/classification_comparison_plot.png')

# =============================================================================
# ΑΝΑΛΥΤΙΚΗ ΑΞΙΟΛΟΓΗΣΗ ΤΟΥ ΝΙΚΗΤΗ(ΑΛΓΟΡΙΘΜΟΥ) ΠΟΥ ΠΡΟΕΚΥΨΕ ΑΠΟ ΤΟ CROSS-VALIDATION
# =============================================================================

# 1. Διαχωρισμός σε Train/Test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Επιλογή του καλύτερου μοντέλου
winner_name = results_df.iloc[0]['Model']
winner_model = best_models[winner_name]

# 3. Εκπαίδευση και Πρόβλεψη
winner_model.fit(X_train, y_train)
y_pred = winner_model.predict(X_test)

# 4. Εκτύπωση Classification Report για τον νικητή
print(f"\n--- ΑΝΑΛΥΤΙΚΟ REPORT ΓΙΑ ΤΟΝ ΝΙΚΗΤΗ: {winner_name} ---")
target_names = ['Insomnia', 'None', 'Sleep Apnea']
print(classification_report(y_test, y_pred, target_names=target_names))

# 5. Δημιουργία Confusion Matrix και Οπτικοποίηση
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.title(f'Confusion Matrix - {winner_name}')
plt.ylabel('Πραγματικό')
plt.xlabel('Πρόβλεψη')
plt.tight_layout()
plt.savefig('output/best_model_confusion_matrix.png')
plt.show()

print(f"\nΌλα τα αποτελέσματα και τα γραφήματα αποθηκεύτηκαν στον φάκελο 'output/'.")