# Sleep Health and Lifestyle Analysis Project

Αυτό το repository περιέχει τον κώδικα και την ανάλυση για την εργασία στο μάθημα της Εξόρυξης Δεδομένων. Στόχος είναι η μελέτη των παραγόντων που επηρεάζουν την ποιότητα ύπνου και η πρόβλεψη διαταραχών ύπνου.

## Δομή Project

Το project χωρίζεται σε 4 κύρια scripts που πρέπει να εκτελεστούν με τη σειρά:

1.  **`scripts/01_preprocessing.py`**: 
    *   Φόρτωση και καθαρισμός δεδομένων.
    *   Διαχείριση ελλιπών τιμών.
    *   Κωδικοποίηση κατηγορικών μεταβλητών (Label Encoding).
    *   Δημιουργία του `data/Sleep_health_cleaned.csv`.

2.  **`scripts/02_classification.py`**: 
    *   Συγκριτική μελέτη 5 αλγορίθμων (Decision Tree, KNN, Naive Bayes, Random Forest, AdaBoost).
    *   Χρήση Cross-Validation και Grid Search για βελτιστοποίηση.
    *   Αξιολόγηση με Accuracy, Precision, Recall, F1-Score.

3.  **`scripts/03_clustering.py`**: 
    *   Ανάλυση συσταδοποίησης (Clustering).
    *   Αλγόριθμοι: K-Means (Elbow Method), Hierarchical (Dendrogram), DBSCAN.
    *   Οπτικοποίηση αποτελεσμάτων.

4.  **`scripts/04_association_rules.py`**: 
    *   Εξόρυξη κανόνων συσχέτισης με τον αλγόριθμο Apriori.
    *   Εντοπισμός παραγόντων κινδύνου για Αϋπνία και Άπνοια.

*   **`scripts/inspect_bins.py`**: Βοηθητικό script για τον έλεγχο της κατανομής των δεδομένων κατά τη διακριτοποίηση.

## Εγκατάσταση & Εκτέλεση

1.  **Εγκατάσταση βιβλιοθηκών:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Εκτέλεση των scripts:**
    ```bash
    python scripts/01_preprocessing.py
    python scripts/02_classification.py
    python scripts/03_clustering.py
    python scripts/04_association_rules.py
    ```

## Αποτελέσματα

Όλα τα αποτελέσματα (γραφήματα, πίνακες, κανόνες) αποθηκεύονται αυτόματα στον φάκελο `output/`.

## Συντάκτης
Μιλτιάδης Μουρτιάς
