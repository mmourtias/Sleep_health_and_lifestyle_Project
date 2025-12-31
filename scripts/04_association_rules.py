import os
import numpy as np
import pandas as pd
import sys

try:
    from mlxtend.frequent_patterns import apriori, association_rules
except Exception:
    print("Missing dependency: mlxtend is not installed. Install with: pip install mlxtend")
    sys.exit(1)

# Set pandas display options to prevent wrapping in terminal output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# =========================
# 1) Load dataset
# =========================
df = pd.read_csv("data/Sleep_health_cleaned.csv")
df.columns = df.columns.str.strip()

# Optional: drop ID if exists
if "Person ID" in df.columns:
    df = df.drop(columns=["Person ID"])

# Try to recover human-readable labels from encoded columns
encoded_cols = [c for c in df.columns if c.endswith('_encoded')]
if encoded_cols:
    try:
        from sklearn.preprocessing import LabelEncoder
        raw_path = os.path.join('data', 'Sleep_health_and_lifestyle_dataset.csv')
        if os.path.exists(raw_path):
            raw = pd.read_csv(raw_path)
            if 'Sleep Disorder' in raw.columns:
                raw['Sleep Disorder'] = raw['Sleep Disorder'].fillna('None')
            # Fix BMI in raw to match preprocessing
            if 'BMI Category' in raw.columns:
                raw['BMI Category'] = raw['BMI Category'].replace('Normal Weight', 'Normal')
                
            for base in ['Gender', 'BMI Category', 'Sleep Disorder', 'Occupation']:
                enc = base + '_encoded'
                if enc in df.columns and base in raw.columns:
                    le = LabelEncoder()
                    le.fit(raw[base].astype(str))
                    # Create mapping dictionary
                    enc_to_label = {int(code): label for code, label in zip(le.transform(le.classes_), le.classes_)}
                    # Apply mapping
                    df[base] = df[enc].map(enc_to_label).astype(str)
    except Exception as e:
        print(f"Warning: Could not decode labels: {e}")
        pass

# =========================
# 2) Discretization (binning)
# =========================
df_ar = df.copy()

# Ensure numeric
num_cols = [
    "Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", 
    "Stress Level", "Heart Rate", "Daily Steps", "Systolic", "Diastolic"
]
for c in num_cols:
    if c in df_ar.columns:
        df_ar[c] = pd.to_numeric(df_ar[c], errors="coerce")

# Columns to use for Association Rules
use_cols = ["Gender", "BMI Category", "Sleep Disorder"] + num_cols
use_cols = [c for c in use_cols if c in df_ar.columns]
df_ar = df_ar.dropna(subset=use_cols).copy()

# --- BINS (Improved Labels) ---
# Using simpler labels because the column name will be prepended later
if "Age" in df_ar.columns:
    df_ar["Age_bin"] = pd.cut(
        df_ar["Age"],
        bins=[-np.inf, 35, 50, np.inf],
        labels=["Young", "Middle", "Old"]
    )

if "Sleep Duration" in df_ar.columns:
    df_ar["SleepDuration_bin"] = pd.cut(
        df_ar["Sleep Duration"],
        bins=[-np.inf, 6, 7, np.inf],
        labels=["Short", "Normal", "Long"],
        right=False
    )

if "Quality of Sleep" in df_ar.columns:
    df_ar["Quality_bin"] = pd.cut(
        df_ar["Quality of Sleep"],
        bins=[-np.inf, 5, 7, np.inf],
        labels=["Poor", "Medium", "Good"]
    )

if "Stress Level" in df_ar.columns:
    df_ar["Stress_bin"] = pd.cut(
        df_ar["Stress Level"],
        bins=[-np.inf, 4, 6, np.inf],
        labels=["Low", "Medium", "High"]
    )

if "Physical Activity Level" in df_ar.columns:
    df_ar["Activity_bin"] = pd.cut(
        df_ar["Physical Activity Level"],
        bins=[-np.inf, 40, 70, np.inf],
        labels=["Low", "Moderate", "High"]
    )

if "Daily Steps" in df_ar.columns:
    df_ar["Steps_bin"] = pd.cut(
        df_ar["Daily Steps"],
        bins=[-np.inf, 5000, 8000, np.inf],
        labels=["Low", "Medium", "High"]
    )

if "Heart Rate" in df_ar.columns:
    df_ar["HR_bin"] = pd.cut(
        df_ar["Heart Rate"],
        bins=[-np.inf, 75, np.inf],
        labels=["Normal", "Elevated"]
    )

if "Systolic" in df_ar.columns and "Diastolic" in df_ar.columns:
    def bp_cat(row):
        s, d = row["Systolic"], row["Diastolic"]
        if (s < 120) and (d < 80):
            return "Normal"
        if (120 <= s < 130) and (d < 80):
            return "Elevated"
        return "High"
    df_ar["BP_bin"] = df_ar.apply(bp_cat, axis=1)

# =========================
# 3) Build transactions
# =========================
final_cols = [
    "Gender", "BMI Category", "Sleep Disorder", 
    "Age_bin", "SleepDuration_bin", "Quality_bin", "Stress_bin", 
    "Activity_bin", "Steps_bin", "HR_bin", "BP_bin"
]
final_cols = [c for c in final_cols if c in df_ar.columns]

# Create tokens: "Column=Value" (Safe approach)
items = df_ar[final_cols].astype(str).apply(lambda s: s.str.strip())
token_df = pd.DataFrame(index=items.index)
for c in final_cols:
    token_df[c] = c + "=" + items[c]

# Merge rare tokens to reduce noise
def merge_rare_tokens(df_tokens, min_count=5):
    df_out = df_tokens.copy()
    for col in df_out.columns:
        counts = df_out[col].value_counts()
        rare = counts[counts < min_count].index
        if len(rare) > 0:
            df_out[col] = df_out[col].replace(rare, f"{col}=Other")
    return df_out

MIN_COUNT_RARE = 5
token_df = merge_rare_tokens(token_df, min_count=MIN_COUNT_RARE)

# One-hot encode
basket = pd.get_dummies(token_df, prefix="", prefix_sep="").astype(bool)

print(f"Transactions shape: {basket.shape}")

# =========================
# 4) Apriori
# =========================
# Using 0.03 as discussed to catch Sleep Apnea patterns
MIN_SUPPORT = 0.03
APRIORI_MAX_LEN = 3

print(f"Running Apriori with min_support={MIN_SUPPORT}...")
freq_itemsets = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True, max_len=APRIORI_MAX_LEN)
print(f"Frequent itemsets found: {len(freq_itemsets)}")

# =========================
# 5) Association Rules
# =========================
MIN_CONFIDENCE = 0.60
MIN_LIFT = 1.10

rules = association_rules(freq_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules[rules["lift"] > MIN_LIFT].copy()

# Helper to format sets
def set_to_str(s):
    return ", ".join(sorted(list(s)))

rules["LHS"] = rules["antecedents"].apply(set_to_str)
rules["RHS"] = rules["consequents"].apply(set_to_str)

# Filter: Keep rules where LHS <= 3 items and RHS == 1 item (cleaner)
rules = rules[
    (rules["antecedents"].apply(len) <= 3) &
    (rules["consequents"].apply(len) == 1)
].copy()

# NEW: Add a key to identify unique rules and drop duplicates.
# This prevents adding the same logical rule twice if it was generated from different source itemsets.
rules["rule_key"] = rules["LHS"] + " -> " + rules["RHS"]
rules = rules.drop_duplicates(subset="rule_key", keep="first")

# Sort
rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)

# =========================
# 6) Targeted Analysis: Sleep Disorder Rules
# =========================
# Filter rules where the consequence is a Sleep Disorder
sleep_disorder_rules = rules[
    rules['RHS'].str.contains('Sleep Disorder=Sleep Apnea') | 
    rules['RHS'].str.contains('Sleep Disorder=Insomnia')
].copy()

# --- Selection Logic (Diversity: Pick best rule for each disorder type first) ---
picked_rules_list = []
seen_indexes = set()

# 1. Pick best rule for each disorder type present (για να έχουμε ποικιλία)
for disorder in sleep_disorder_rules['RHS'].unique():
    subset = sleep_disorder_rules[sleep_disorder_rules['RHS'] == disorder]
    if not subset.empty:
        best_rule = subset.iloc[0]
        picked_rules_list.append(best_rule)
        seen_indexes.add(best_rule.name) # .name is the index label

# 2. Fill up to 12 with remaining best rules
target_count = 12
# Use the index of the dataframe to exclude seen rules
remaining = sleep_disorder_rules.loc[~sleep_disorder_rules.index.isin(seen_indexes)]
if len(picked_rules_list) < target_count:
    needed = target_count - len(picked_rules_list)
    others = remaining.head(needed)
    for _, rule in others.iterrows():
        picked_rules_list.append(rule)

picked_df = pd.DataFrame(picked_rules_list)
if not picked_df.empty:
    # Final sort and deduplication just in case the logic picked overlapping rules
    picked_df["rule_key"] = picked_df["LHS"] + " -> " + picked_df["RHS"]
    picked_df = picked_df.drop_duplicates(subset="rule_key", keep="first")
    picked_df = picked_df.sort_values(by=['lift', 'confidence'], ascending=False).reset_index(drop=True)
    
# =========================
# 7) Save Outputs
# =========================
os.makedirs("output", exist_ok=True)

cols_out = ["support", "confidence", "lift", "LHS", "RHS"]

# Save all rules
rules[cols_out].to_csv("output/apriori_rules_full.csv", index=False)

# Save top 30 general rules
top_rules = rules[cols_out].head(30)
top_rules.to_csv("output/apriori_rules_top30.csv", index=False)

# Save all targeted sleep disorder rules
sleep_disorder_rules[cols_out].to_csv("output/apriori_rules_sleep_disorders.csv", index=False)
if not picked_df.empty:
    # Save the final selection for the report
    picked_df[cols_out].head(12).to_csv("output/apriori_rules_for_report.csv", index=False)

print("\n" + "="*80)
print("TOP 12 RULES PREDICTING SLEEP DISORDERS (Sorted by Lift)")
print("="*80)

if not picked_df.empty:
    # Create a clean table for display
    display_df = picked_df[["LHS", "RHS", "support", "confidence", "lift"]].head(12).copy() # Ensure we only take the top 12
    # Round for better display
    display_df["support"] = display_df["support"].round(3)
    display_df["confidence"] = display_df["confidence"].round(3)
    display_df["lift"] = display_df["lift"].round(2)

    # Add the rule number column at the beginning
    display_df.insert(0, 'Κανόνας', [f"Κανόνας {i+1}" for i in range(len(display_df))])

    print(display_df.to_string(index=False))
else:
    print("No strong rules found specifically predicting Sleep Disorders with current thresholds.")

print("\nSaved files in 'output/':")
print(" - apriori_rules_full.csv")
print(" - apriori_rules_top30.csv")
print(" - apriori_rules_sleep_disorders.csv")
print(" - apriori_rules_for_report.csv (Use this for your conclusions!)")
