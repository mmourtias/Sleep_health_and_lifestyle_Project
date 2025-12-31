import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Load cleaned data
df = pd.read_csv('data/Sleep_health_cleaned.csv')

# Try to map encoded back to labels
raw_path = 'data/Sleep_health_and_lifestyle_dataset.csv'
if os.path.exists(raw_path):
    raw = pd.read_csv(raw_path)
    if 'Sleep Disorder' in raw.columns:
        raw['Sleep Disorder'] = raw['Sleep Disorder'].fillna('None')
    for base in ['Gender', 'BMI Category', 'Sleep Disorder']:
        enc = base + '_encoded'
        if enc in df.columns and base in raw.columns:
            le = LabelEncoder()
            le.fit(raw[base].astype(str))
            enc_to_label = {int(code): label for code, label in zip(le.transform(le.classes_), le.classes_)}
            df[base] = df[enc].map(enc_to_label).astype(str)

# Numeric columns
num_cols = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps", "Systolic", "Diastolic"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

use_cols = ['Gender', 'BMI Category', 'Sleep Disorder'] + [c for c in num_cols if c in df.columns]

# Drop rows with missing in used cols
df = df.dropna(subset=use_cols).copy()

# Binning
df['Age_bin'] = pd.cut(df['Age'], bins=[-np.inf, 35, 50, np.inf], labels=['Age=Young', 'Age=Middle', 'Age=Old'])

df['SleepDuration_bin'] = pd.cut(df['Sleep Duration'], bins=[-np.inf, 6, 7, np.inf], labels=['Sleep=Short', 'Sleep=Normal', 'Sleep=Long'], right=False)

df['Quality_bin'] = pd.cut(df['Quality of Sleep'], bins=[-np.inf, 5, 7, np.inf], labels=['Quality=Poor', 'Quality=Medium', 'Quality=Good'])

df['Stress_bin'] = pd.cut(df['Stress Level'], bins=[-np.inf, 4, 6, np.inf], labels=['Stress=Low', 'Stress=Medium', 'Stress=High'])

df['Activity_bin'] = pd.cut(df['Physical Activity Level'], bins=[-np.inf, 40, 70, np.inf], labels=['Activity=Low', 'Activity=Moderate', 'Activity=High'])

df['Steps_bin'] = pd.cut(df['Daily Steps'], bins=[-np.inf, 5000, 8000, np.inf], labels=['Steps=Low', 'Steps=Medium', 'Steps=High'])

if 'Heart Rate' in df.columns:
    df['HR_bin'] = pd.cut(df['Heart Rate'], bins=[-np.inf, 75, np.inf], labels=['HR=Normal', 'HR=Elevated'])

if 'Systolic' in df.columns and 'Diastolic' in df.columns:
    def bp_cat(row):
        s, d = row['Systolic'], row['Diastolic']
        if (s < 120) and (d < 80):
            return 'BP=Normal'
        if (120 <= s < 130) and (d < 80):
            return 'BP=Elevated'
        return 'BP=High'
    df['BP_bin'] = df.apply(bp_cat, axis=1)

item_cols = [c for c in ['Gender', 'BMI Category', 'Sleep Disorder', 'Age_bin', 'SleepDuration_bin', 'Quality_bin', 'Stress_bin', 'Activity_bin', 'Steps_bin', 'HR_bin', 'BP_bin'] if c in df.columns]
items = df[item_cols].astype(str).apply(lambda s: s.str.strip())

# Summary
N = len(df)
print(f'Total rows after dropna: {N}')

for c in item_cols:
    print('\nColumn:', c)
    print(items[c].value_counts(dropna=False).to_string())

# Token counts
tokens = items.stack().value_counts().sort_values(ascending=False)
print('\nTop 40 tokens:')
print(tokens.head(40).to_string())

small = tokens[tokens < 5]
print(f'\nTokens with count <5: {len(small)}')
if len(small) > 0:
    print(small.sort_values().to_string())

# Suggested supports
print('\nSuggested min_support for min_count=5:', 5.0 / N)
print('Suggested min_support for min_count=10:', 10.0 / N)
