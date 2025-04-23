# =====================================
#  Class: FlexiblePatternMiner (Improved with Comments)
# Description: Modular pattern mining engine that takes raw data,
# encodes transactions, computes recency, mines frequent patterns, and scores rules.
# =====================================

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import MinMaxScaler

class FlexiblePatternMiner:
    def __init__(self, raw_df, user_col='user_id', item_col='item', date_col='date'):
        print("Initializing FlexiblePatternMiner...")
        # Store raw transaction data and column names
        self.raw_df = raw_df.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.date_col = date_col

        # Process input: encode transactions and compute recency scores
        self.df_encoded = self._encode_transactions()
        self.recency_score = self._compute_recency_scores()

        # Default configuration
        self.min_support = 0.002
        self.min_confidence = 0.04
        self.selected_algorithms = ['Apriori', 'FP-Growth']
        self.weights = (0.05, 0.05, 0.70, 0.20)  # (support, confidence, lift, recency)

        # Initialize storage and algorithm options
        self.algorithms = {'Apriori': apriori, 'FP-Growth': fpgrowth}
        self.frequent_itemsets = pd.DataFrame()
        self.rules_df = pd.DataFrame()

    def _encode_transactions(self):
        # Group items by user and convert to list of lists
        grouped = self.raw_df.groupby(self.user_col)[self.item_col].apply(list).tolist()
        # One-hot encode the transactions
        encoder = TransactionEncoder()
        encoded_array = encoder.fit_transform(grouped)
        return pd.DataFrame(encoded_array, columns=encoder.columns_)

    def _compute_recency_scores(self):
        # Find most recent transaction date overall
        most_recent = self.raw_df[self.date_col].max()
        # Compute recency: days since each item last occurred
        recency_dict = self.raw_df.groupby(self.item_col)[self.date_col] \
            .max().apply(lambda x: (most_recent - x).days).to_dict()
        # Normalize recency values to [0, 1]
        scaler = MinMaxScaler()
        values = np.array(list(recency_dict.values())).reshape(-1, 1)
        scaled = scaler.fit_transform(values).flatten()
        return dict(zip(recency_dict.keys(), scaled))

    def _avg_recency(self, itemset):
        # Compute average recency score for an itemset
        return np.mean([self.recency_score.get(item, 0) for item in itemset])

    def _rule_recency_score(self, row):
        # Compute average recency score for a rule (antecedents + consequents)
        items = list(row['antecedents']) + list(row['consequents'])
        return np.mean([self.recency_score.get(item, 0) for item in items])

    def set_min_support(self, support):
        print(f"Setting minimum support to {support}")
        self.min_support = support

    def set_min_confidence(self, confidence):
        print(f"Setting minimum confidence to {confidence}")
        self.min_confidence = confidence

    def set_weights(self, alpha, beta, gamma, delta):
        print(f"Setting weights: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")
        self.weights = (alpha, beta, gamma, delta)

    def set_selected_algorithms(self, algos):
        print(f"Setting selected algorithms: {algos}")
        self.selected_algorithms = algos

    def get_rules(self):
        # Return DataFrame of generated rules
        return self.rules_df

    def get_frequent_itemsets(self):
        # Return DataFrame of frequent itemsets
        return self.frequent_itemsets

    def mine_frequent_itemsets(self):
        print("Mining frequent itemsets...")
        all_itemsets = []
        # Loop through each selected algorithm
        for name in self.selected_algorithms:
            print(f"Running {name}...")
            func = self.algorithms[name]  # Get function (apriori or fpgrowth)
            itemsets = func(self.df_encoded, min_support=self.min_support, use_colnames=True).copy()
            itemsets['algorithm'] = name  # Tag source algorithm
            itemsets['length'] = itemsets['itemsets'].apply(lambda x: len(x))  # Store itemset length
            itemsets['recency_score'] = itemsets['itemsets'].apply(self._avg_recency)  # Add recency
            all_itemsets.append(itemsets)
        # Combine itemsets from all algorithms
        self.frequent_itemsets = pd.concat(all_itemsets, ignore_index=True)
        print(f"Total itemsets mined: {len(self.frequent_itemsets)}")

    def generate_rules(self):
        print("Generating association rules...")
        all_rules = []
        # Generate rules per algorithm
        for algo in self.selected_algorithms:
            print(f"Generating rules for {algo}...")
            itemsets_algo = self.frequent_itemsets[self.frequent_itemsets['algorithm'] == algo]
            rules = association_rules(itemsets_algo, metric="confidence", min_threshold=0.001)
            rules = rules[rules['confidence'] >= self.min_confidence]
            if rules.empty:
                print(f"No rules for {algo} above confidence {self.min_confidence}")
                continue
            rules['algorithm'] = algo
            rules['recency_score'] = rules.apply(self._rule_recency_score, axis=1)
            all_rules.append(rules)
        # Combine rule sets
        self.rules_df = pd.concat(all_rules, ignore_index=True) if all_rules else pd.DataFrame()
        print(f"Total rules after filtering: {len(self.rules_df)}")

    def apply_composite_scoring(self):
        print("Applying composite scoring...")
        if self.rules_df.empty:
            print("No rules available to score.")
            return
        scaler = MinMaxScaler()
        # Normalize each metric
        for metric in ['support', 'confidence', 'lift', 'recency_score']:
            self.rules_df[f'{metric}_norm'] = scaler.fit_transform(self.rules_df[[metric]])
        alpha, beta, gamma, delta = self.weights
        # Weighted sum of normalized metrics
        self.rules_df['composite_score_with_recency'] = (
            alpha * self.rules_df['support_norm'] +
            beta * self.rules_df['confidence_norm'] +
            gamma * self.rules_df['lift_norm'] +
            delta * self.rules_df['recency_score_norm']
        )
        print("Scoring complete.")

    def get_top_rules(self, top_n=10):
        print(f"Retrieving top {top_n} rules per algorithm...")
        if self.rules_df.empty:
            print("No rules available.")
            return pd.DataFrame()
        # Get top N rules for Apriori and FP-Growth based on composite score
        top_apriori = self.rules_df[self.rules_df['algorithm'] == 'Apriori']\
            .sort_values(by='composite_score_with_recency', ascending=False).head(top_n)
        top_fp = self.rules_df[self.rules_df['algorithm'] == 'FP-Growth']\
            .sort_values(by='composite_score_with_recency', ascending=False).head(top_n)
        # Combine and return top rules
        combined = pd.concat([top_apriori, top_fp], ignore_index=True)[
            ['algorithm', 'antecedents', 'consequents', 'composite_score_with_recency']
        ]
        print("Top rules retrieval complete.")
        return combined

    def export_rules(self, prefix="rules"):
        print(f"Exporting rules with prefix '{prefix}'...")
        export_cols = [
            'algorithm', 'antecedents', 'consequents', 'support', 'confidence',
            'lift', 'recency_score', 'composite_score_with_recency'
        ]
        # Export full, apriori, and FP-Growth rules to separate CSV files
        self.rules_df[export_cols].to_csv(f"{prefix}_combined.csv", index=False)
        self.rules_df[self.rules_df['algorithm'] == 'Apriori'][export_cols].to_csv(f"{prefix}_apriori.csv", index=False)
        self.rules_df[self.rules_df['algorithm'] == 'FP-Growth'][export_cols].to_csv(f"{prefix}_fpgrowth.csv", index=False)
        print("Rules exported to CSV.")
