import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# 1. Φόρτωση των δεδομένων
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')

# 2. Διόρθωση του BMI Category (Συγχώνευση 'Normal Weight' σε 'Normal')
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

# 3. Διαχείριση των κενών στο Sleep Disorder (NaN -> None)
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# 4. Διαχωρισμός της Αρτηριακής Πίεσης
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

# 5. Μετατροπή Κατηγορικών σε Αριθμητικά (Label Encoding)
le = LabelEncoder()
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']

for col in categorical_cols:
    df[col + '_encoded'] = le.fit_transform(df[col])
    # Εκτύπωση για να ξέρουμε ποιος αριθμός αντιστοιχεί σε τι
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Mapping for {col}: {mapping}")

# 6. Αφαίρεση περιττών στηλών
# Κρατάμε μόνο τις _encoded εκδόσεις και τις αριθμητικές
cols_to_drop = ['Person ID', 'Blood Pressure', 'Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
df_final = df.drop(columns=cols_to_drop)



# 7. Αποθήκευση του καθαρού αρχείου
if not os.path.exists('data'):
    os.makedirs('data')
df_final.to_csv('data/Sleep_health_cleaned.csv', index=False)

print("\nΤο dataset είναι πλέον πλήρως καθαρισμένο και αποθηκεύτηκε ως 'data/Sleep_health_cleaned.csv'")
print(df_final.head())