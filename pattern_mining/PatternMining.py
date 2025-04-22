# =====================================
#  Class: FlexiblePatternMiner
# Description: Modular, configurable mining pipeline with setter/getter methods.
# Supports Apriori/FP-Growth, min_support/confidence, scoring weights, and recency.
# =====================================

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.preprocessing import MinMaxScaler

class FlexiblePatternMiner:
    def __init__(self, df_encoded, recency_score):
        # Initialize with encoded transaction dataframe and recency scores
        print("Initializing FlexiblePatternMiner...")
        self.df_encoded = df_encoded  # One-hot encoded transactional data
        self.recency_score = recency_score  # Dictionary of recency values
        self.min_support = 0.005  # Default support threshold
        self.min_confidence = 0.04  # Default confidence threshold
        self.algorithms = {'Apriori': apriori, 'FP-Growth': fpgrowth}  # Mapping of algorithm names to functions
        self.selected_algorithms = ['Apriori', 'FP-Growth']  # Algorithms selected to run
        self.weights = (0.2, 0.2, 0.4, 0.2)  # Default weights: (support, confidence, lift, recency)
        self.frequent_itemsets = pd.DataFrame()  # Will hold all itemsets
        self.rules_df = pd.DataFrame()  # Will hold final association rules

    # === SETTERS ===
    def set_min_support(self, support):
        # Set the minimum support threshold
        print(f"Setting minimum support to {support}")
        self.min_support = support

    def set_min_confidence(self, confidence):
        # Set the minimum confidence threshold
        print(f"Setting minimum confidence to {confidence}")
        self.min_confidence = confidence

    def set_weights(self, alpha, beta, gamma, delta):
        # Set custom weights for composite scoring
        print(f"Setting weights: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")
        self.weights = (alpha, beta, gamma, delta)

    def set_selected_algorithms(self, algos):
        # Set which algorithms to run (Apriori, FP-Growth)
        print(f"Setting selected algorithms: {algos}")
        self.selected_algorithms = algos

    # === GETTERS ===
    def get_rules(self):
        # Return all association rules
        return self.rules_df

    def get_frequent_itemsets(self):
        # Return mined frequent itemsets
        return self.frequent_itemsets

    # === INTERNAL HELPERS ===
    def _avg_recency(self, itemset):
        # Compute average recency score for a given itemset
        return np.mean([self.recency_score.get(item, 0) for item in itemset])

    def _rule_recency_score(self, row):
        # Compute recency score for a full rule (antecedent + consequent)
        items = list(row['antecedents']) + list(row['consequents'])
        return np.mean([self.recency_score.get(item, 0) for item in items])

    # === CORE METHODS ===
    def mine_frequent_itemsets(self):
        # Mine itemsets for each selected algorithm
        print("Mining frequent itemsets...")
        all_itemsets = []
        for name in self.selected_algorithms:
            print(f"Running {name}...")
            func = self.algorithms[name]  # apriori or fpgrowth function
            itemsets = func(self.df_encoded, min_support=self.min_support, use_colnames=True).copy()
            itemsets['algorithm'] = name  # Tag which algorithm produced the itemsets
            itemsets['length'] = itemsets['itemsets'].apply(lambda x: len(x))  # Length of itemset
            itemsets['recency_score'] = itemsets['itemsets'].apply(self._avg_recency)  # Add recency to itemset
            all_itemsets.append(itemsets)
        self.frequent_itemsets = pd.concat(all_itemsets, ignore_index=True)  # Combine all itemsets
        print(f"Total itemsets mined: {len(self.frequent_itemsets)}")

    def generate_rules(self):
        # Generate association rules based on mined itemsets
        print("Generating association rules...")
        all_rules = []
        for algo in self.selected_algorithms:
            print(f"Generating rules for {algo}...")
            itemsets_algo = self.frequent_itemsets[self.frequent_itemsets['algorithm'] == algo]  # Filter itemsets
            rules = association_rules(itemsets_algo, metric="confidence", min_threshold=0.001)  # Generate raw rules
            rules = rules[rules['confidence'] >= self.min_confidence]  # Filter based on confidence
            if rules.empty:
                print(f"No rules for {algo} above confidence {self.min_confidence}")
                continue
            rules['algorithm'] = algo  # Add algorithm name to rules
            rules['recency_score'] = rules.apply(self._rule_recency_score, axis=1)  # Compute recency
            all_rules.append(rules)
        self.rules_df = pd.concat(all_rules, ignore_index=True) if all_rules else pd.DataFrame()
        print(f"Total rules after filtering: {len(self.rules_df)}")

    def apply_composite_scoring(self):
        # Normalize and apply weighted composite score to rules
        print("Applying composite scoring...")
        if self.rules_df.empty:
            print("No rules available to score.")
            return
        scaler = MinMaxScaler()  # Use MinMax scaling to normalize [0,1]
        for metric in ['support', 'confidence', 'lift', 'recency_score']:
            self.rules_df[f'{metric}_norm'] = scaler.fit_transform(self.rules_df[[metric]])
        alpha, beta, gamma, delta = self.weights  # Extract weights
        self.rules_df['composite_score_with_recency'] = (
            alpha * self.rules_df['support_norm'] +
            beta * self.rules_df['confidence_norm'] +
            gamma * self.rules_df['lift_norm'] +
            delta * self.rules_df['recency_score_norm']
        )
        print("Scoring complete.")

    def get_top_rules(self, top_n=10):
        # Get top N rules for each algorithm based on composite score
        print(f"Retrieving top {top_n} rules per algorithm...")
        if self.rules_df.empty:
            print("No rules available.")
            return pd.DataFrame()
        top_apriori = self.rules_df[self.rules_df['algorithm'] == 'Apriori']\
            .sort_values(by='composite_score_with_recency', ascending=False).head(top_n)
        top_fp = self.rules_df[self.rules_df['algorithm'] == 'FP-Growth']\
            .sort_values(by='composite_score_with_recency', ascending=False).head(top_n)
        combined = pd.concat([top_apriori, top_fp], ignore_index=True)[
            ['algorithm', 'antecedents', 'consequents', 'composite_score_with_recency']
        ]
        print("Top rules retrieval complete.")
        return combined

    def export_rules(self, prefix="rules"):
        # Export rules to CSV with selected columns
        print(f"Exporting rules with prefix '{prefix}'...")
        export_cols = [
            'algorithm', 'antecedents', 'consequents', 'support', 'confidence',
            'lift', 'recency_score', 'composite_score_with_recency'
        ]
        self.rules_df[export_cols].to_csv(f"{prefix}_combined.csv", index=False)
        self.rules_df[self.rules_df['algorithm'] == 'Apriori'][export_cols].to_csv(f"{prefix}_apriori.csv", index=False)
        self.rules_df[self.rules_df['algorithm'] == 'FP-Growth'][export_cols].to_csv(f"{prefix}_fpgrowth.csv", index=False)
        print("Rules exported to CSV.")
