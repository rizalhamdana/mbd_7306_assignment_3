# Save the updated class into a Python file that can be used as a module
module_code = """
# =====================================
#  Class: FlexiblePatternMiner (Production-Ready)
# Description: Modular pattern mining engine for system integration.
# Supports Apriori/FP-Growth, configurable scoring, export, and composite ranking.
# =====================================

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import MinMaxScaler

class FlexiblePatternMiner:
    def __init__(self, raw_df, user_col='user_id', item_col='item', date_col='date'):
        self.raw_df = raw_df.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.date_col = date_col
        self.df_encoded = self._encode_transactions()
        self.recency_score = self._compute_recency_scores()
        self.min_support = 0.002
        self.min_confidence = 0.04
        self.selected_algorithms = ['Apriori', 'FP-Growth']
        self.weights = (0.05, 0.05, 0.70, 0.20)
        self.algorithms = {'Apriori': apriori, 'FP-Growth': fpgrowth}
        self.frequent_itemsets = pd.DataFrame()
        self.rules_df = pd.DataFrame()

    def _encode_transactions(self):
        grouped = self.raw_df.groupby(self.user_col)[self.item_col].apply(list).tolist()
        encoder = TransactionEncoder()
        encoded_array = encoder.fit_transform(grouped)
        return pd.DataFrame(encoded_array, columns=encoder.columns_)

    def _compute_recency_scores(self):
        most_recent = self.raw_df[self.date_col].max()
        recency_dict = self.raw_df.groupby(self.item_col)[self.date_col].max().apply(lambda x: (most_recent - x).days).to_dict()
        scaler = MinMaxScaler()
        values = np.array(list(recency_dict.values())).reshape(-1, 1)
        scaled = scaler.fit_transform(values).flatten()
        return dict(zip(recency_dict.keys(), scaled))

    def _avg_recency(self, itemset):
        return np.mean([self.recency_score.get(item, 0) for item in itemset])

    def _rule_recency_score(self, row):
        items = list(row['antecedents']) + list(row['consequents'])
        return np.mean([self.recency_score.get(item, 0) for item in items])

    def set_min_support(self, support): self.min_support = support
    def set_min_confidence(self, confidence): self.min_confidence = confidence
    def set_weights(self, alpha, beta, gamma, delta): self.weights = (alpha, beta, gamma, delta)
    def set_selected_algorithms(self, algos): self.selected_algorithms = algos
    def get_rules(self): return self.rules_df
    def get_frequent_itemsets(self): return self.frequent_itemsets

    def mine_frequent_itemsets(self):
        all_itemsets = []
        for name in self.selected_algorithms:
            func = self.algorithms[name]
            df_binary_only = self.df_encoded.drop(columns=['recency_score'], errors='ignore')
            itemsets = func(df_binary_only, min_support=self.min_support, use_colnames=True).copy()
            itemsets['algorithm'] = name
            itemsets['length'] = itemsets['itemsets'].apply(lambda x: len(x))
            itemsets['recency_score'] = itemsets['itemsets'].apply(self._avg_recency)
            all_itemsets.append(itemsets)
        self.frequent_itemsets = pd.concat(all_itemsets, ignore_index=True)

    def generate_rules(self):
        all_rules = []
        for algo in self.selected_algorithms:
            itemsets_algo = self.frequent_itemsets[self.frequent_itemsets['algorithm'] == algo]
            rules = association_rules(itemsets_algo, metric="confidence", min_threshold=0.001)
            rules = rules[rules['confidence'] >= self.min_confidence]
            if rules.empty:
                continue
            rules['algorithm'] = algo
            rules['recency_score'] = rules.apply(self._rule_recency_score, axis=1)
            all_rules.append(rules)
        self.rules_df = pd.concat(all_rules, ignore_index=True) if all_rules else pd.DataFrame()

    def apply_composite_scoring(self):
        if self.rules_df.empty:
            return
        scaler = MinMaxScaler()
        for metric in ['support', 'confidence', 'lift', 'recency_score']:
            self.rules_df[f'{metric}_norm'] = scaler.fit_transform(self.rules_df[[metric]])
        alpha, beta, gamma, delta = self.weights
        self.rules_df['composite_score_with_recency'] = (
            alpha * self.rules_df['support_norm'] +
            beta * self.rules_df['confidence_norm'] +
            gamma * self.rules_df['lift_norm'] +
            delta * self.rules_df['recency_score_norm']
        )

    def get_top_rules(self, top_n=10):
        if self.rules_df.empty:
            return pd.DataFrame()
        top_apriori = self.rules_df[self.rules_df['algorithm'] == 'Apriori']\
            .sort_values(by='composite_score_with_recency', ascending=False).head(top_n)
        top_fp = self.rules_df[self.rules_df['algorithm'] == 'FP-Growth']\
            .sort_values(by='composite_score_with_recency', ascending=False).head(top_n)
        return pd.concat([top_apriori, top_fp], ignore_index=True)[
            ['algorithm', 'antecedents', 'consequents', 'composite_score_with_recency']
        ]

    def export_rules(self, prefix="rules"):
        export_cols = [
            'algorithm', 'antecedents', 'consequents', 'support', 'confidence',
            'lift', 'recency_score', 'composite_score_with_recency'
        ]
        self.rules_df[export_cols].to_csv(f"{prefix}_combined.csv", index=False)
        self.rules_df[self.rules_df['algorithm'] == 'Apriori'][export_cols].to_csv(f"{prefix}_apriori.csv", index=False)
        self.rules_df[self.rules_df['algorithm'] == 'FP-Growth'][export_cols].to_csv(f"{prefix}_fpgrowth.csv", index=False)
"""

# Save to file
with open("/mnt/data/flexible_pattern_miner.py", "w") as file:
    file.write(module_code)

"/mnt/data/flexible_pattern_miner.py"
