
import pandas as pd
import os
from datetime import datetime
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

class PatternMiningEngine:
    def __init__(self, df_encoded):
        self.df_encoded = df_encoded
        self.min_support = 0.001
        self.min_confidence = 0.001
        self.top_n = 100
        self.weights = (0.34, 0.06, 0.60)
        self.algorithm = 'both'
        print("Initialized PatternMiningEngine with default parameters.")

    def __str__(self):
        return (f"PatternMiningEngine(algorithm='{self.algorithm}', min_support={self.min_support}, "
                f"min_confidence={self.min_confidence}, top_n={self.top_n}, weights={self.weights})")

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, PatternMiningEngine):
            return NotImplemented
        return (
            self.min_support == other.min_support and
            self.min_confidence == other.min_confidence and
            self.top_n == other.top_n and
            self.weights == other.weights and
            self.algorithm == other.algorithm
        )

    def __hash__(self):
        return hash((self.min_support, self.min_confidence, self.top_n, self.weights, self.algorithm))

    def set_min_support(self, value):
        self.min_support = value
        print(f"min_support set to: {value}")

    def set_min_confidence(self, value):
        self.min_confidence = value
        print(f"min_confidence set to: {value}")

    def set_top_n(self, value):
        self.top_n = value
        print(f"top_n set to: {value}")

    def set_weights(self, weights_tuple):
        if len(weights_tuple) == 3:
            self.weights = weights_tuple
            print(f"weights set to: {weights_tuple}")

    def set_algorithm(self, algorithm):
        if algorithm in ['apriori', 'fp-growth', 'both']:
            self.algorithm = algorithm
            print(f"algorithm set to: {algorithm}")

    def get_min_support(self):
        return self.min_support

    def get_min_confidence(self):
        return self.min_confidence

    def get_top_n(self):
        return self.top_n

    def get_weights(self):
        return self.weights

    def get_algorithm(self):
        return self.algorithm

    def mine_frequent_itemsets(self):
        print("Mining frequent itemsets...")
        if self.algorithm in ['apriori', 'both']:
            print("Running Apriori...")
            self.frequent_ap = apriori(self.df_encoded, min_support=self.min_support, use_colnames=True)
        if self.algorithm in ['fp-growth', 'both']:
            print("Running FP-Growth...")
            self.frequent_fp = fpgrowth(self.df_encoded, min_support=self.min_support, use_colnames=True)
        print("Frequent itemset mining complete.")

    def generate_rules(self):
        print("Generating association rules...")
        if hasattr(self, 'frequent_ap'):
            print("Generating rules from Apriori itemsets...")
            self.apriori_rules = association_rules(self.frequent_ap, metric="confidence", min_threshold=self.min_confidence)
        if hasattr(self, 'frequent_fp'):
            print("Generating rules from FP-Growth itemsets...")
            self.fpgrowth_rules = association_rules(self.frequent_fp, metric="confidence", min_threshold=self.min_confidence)
        print("Association rule generation complete.")

    def add_c4_and_get_top_rules(self, rules_df, algorithm_name='Apriori'):
        print(f"Scoring rules with Composite C4 for {algorithm_name}...")
        df = rules_df.copy()
        w_supp, w_conf, w_lift = self.weights
        df['composite_c4'] = w_supp * df['support'] + w_conf * df['confidence'] + w_lift * df['lift']
        df['algorithm'] = algorithm_name
        df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(sorted(map(str, list(x)))))
        df['consequents'] = df['consequents'].apply(lambda x: ', '.join(sorted(map(str, list(x)))))
        print(f"Top {self.top_n} rules selected.")
        return df.sort_values(by='composite_c4', ascending=False).head(self.top_n)

    def safe_export(self, df, filename):
        if os.path.exists(filename):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filename.replace('.csv', '')}_backup_{timestamp}.csv"
            os.rename(filename, backup_name)
            print(f"Backup created: {backup_name}")
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")

    def run(self):
        print("Starting pattern mining engine...")
        self.mine_frequent_itemsets()
        self.generate_rules()
        result_frames = []
        if hasattr(self, 'apriori_rules'):
            print("Processing Apriori rules...")
            top_apriori = self.add_c4_and_get_top_rules(self.apriori_rules, 'Apriori')
            self.safe_export(top_apriori, "decided_apriori.csv")
            result_frames.append(top_apriori)
        if hasattr(self, 'fpgrowth_rules'):
            print("Processing FP-Growth rules...")
            top_fpgrowth = self.add_c4_and_get_top_rules(self.fpgrowth_rules, 'FP-Growth')
            self.safe_export(top_fpgrowth, "decided_fpgrowth_rules_c4_lower_threshold.csv")
            result_frames.append(top_fpgrowth)
        if len(result_frames) == 2:
            print("Combining Apriori and FP-Growth results...")
            combined = pd.concat(result_frames, ignore_index=True)
            self.safe_export(combined, "decided_combined_rules_c4_lower_threshold.csv")
            print("Pattern mining complete.")
            return result_frames[0], result_frames[1], combined
        elif len(result_frames) == 1:
            print("Single algorithm result returned.")
            return result_frames[0], None, result_frames[0]
        else:
            print("No rules were generated.")
            return None, None, None
