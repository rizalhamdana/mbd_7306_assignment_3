import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Literal, Tuple, List, Optional

class AssociationRuleEngine:
    def __init__(self, df_encoded: pd.DataFrame):
        """
        Initialize the engine with one-hot encoded transaction data.
        Parameters:
        - df_encoded (pd.DataFrame): One-hot encoded dataset where each column is a unique item.
        """
        self.df_encoded = df_encoded.copy()  # Make a copy to avoid modifying the original dataset
        self.min_support = 0.001  # Minimum support for generating frequent itemsets
        self.min_confidence = 0.001  # Minimum confidence for generating association rules
        self.top_n = 100  # Number of top rules to return after sorting by composite score
        self.weights = (0.2, 0.1, 0.5, 0.2)  # Weights for composite score calculation (alpha, beta, gamma, delta)
        self.algorithm = 'apriori'  # Default algorithm for rule generation: 'apriori' or 'fp_growth'
        self.rules_df = pd.DataFrame()  # DataFrame to hold the generated rules

    # Setter methods to allow customization of parameters
    def set_min_support(self, support: float):
        """Set the minimum support for generating frequent itemsets."""
        self.min_support = support

    def set_min_confidence(self, confidence: float):
        """Set the minimum confidence for generating association rules."""
        self.min_confidence = confidence

    def set_algorithm(self, algorithm: Literal['apriori', 'fp_growth']):
        """Set the algorithm for rule generation: 'apriori' or 'fp_growth'."""
        self.algorithm = algorithm

    def set_weights(self, alpha: float, beta: float, gamma: float, delta: float):
        """
        Set the weights for the composite score calculation (support, confidence, lift, recency).
        Parameters:
        - alpha, beta, gamma, delta: Weights for support, confidence, lift, and recency respectively.
        """
        self.weights = (alpha, beta, gamma, delta)

    # Getter method to retrieve the generated rules
    def get_rules(self) -> pd.DataFrame:
        """Return the DataFrame containing the generated association rules."""
        return self.rules_df

    # Rule mining using either Apriori or FP-Growth
    def mine_rules(self):
        """
        Mine the association rules from the encoded data using the selected algorithm.
        Uses Apriori or FP-Growth depending on the 'self.algorithm' value.
        """
        if self.algorithm == 'apriori':
            frequent = apriori(self.df_encoded, min_support=self.min_support, use_colnames=True)
        else:
            frequent = fpgrowth(self.df_encoded, min_support=self.min_support, use_colnames=True)

        # Generate rules from frequent itemsets using the minimum confidence threshold
        rules = association_rules(frequent, metric="confidence", min_threshold=self.min_confidence)
        rules['algorithm'] = self.algorithm.title()  # Add algorithm type to the rules
        self.rules_df = rules  # Store the generated rules in the DataFrame

    # Normalize support, confidence, lift, and recency score using MinMaxScaler
    def normalize_metrics(self):
        """
        Normalize the support, confidence, lift, and recency_score metrics using MinMaxScaler.
        This will scale the metrics to the range [0, 1] for easier comparison.
        """
        metrics = ['support', 'confidence', 'lift', 'recency_score']
        scaler = MinMaxScaler()
        for metric in metrics:
            if metric in self.rules_df.columns:
                self.rules_df[f'{metric}_norm'] = scaler.fit_transform(self.rules_df[[metric]])

    # Compute the composite score based on the weighted sum of normalized metrics
    def compute_composite_score(self):
        """
        Compute the composite score (C4) for each rule based on normalized support, confidence, lift, and recency.
        The composite score is a weighted sum of these metrics.
        """
        alpha, beta, gamma, delta = self.weights
        self.rules_df['composite_score'] = (
            alpha * self.rules_df.get('support_norm', 0) +
            beta * self.rules_df.get('confidence_norm', 0) +
            gamma * self.rules_df.get('lift_norm', 0) +
            delta * self.rules_df.get('recency_score_norm', 0)
        )

    # Filter the top N rules based on their composite score
    def filter_top_rules(self) -> pd.DataFrame:
        """
        Sort the rules by composite score and return the top N rules.
        """
        return self.rules_df.sort_values(by='composite_score', ascending=False).head(self.top_n)

    # Evaluate precision, recall, and F1 score for rule-based predictions
    def evaluate_predictions(self, y_true: List[set], y_pred: List[set]) -> dict:
        """
        Calculate precision, recall, and F1-score for rule-based predictions.
        Parameters:
        - y_true: List of actual items in the test baskets.
        - y_pred: List of predicted items based on the generated rules.
        Returns a dictionary with precision, recall, and F1-score.
        """
        y_true_flat = [item for sublist in y_true for item in sublist]
        y_pred_flat = [item for sublist in y_pred for item in sublist]
        y_true_bin = [1 if item in y_pred_flat else 0 for item in y_true_flat]
        y_pred_bin = [1] * len(y_true_bin)
        return {
            'precision': precision_score(y_true_bin, y_pred_bin, zero_division=0),
            'recall': recall_score(y_true_bin, y_pred_bin, zero_division=0),
            'f1': f1_score(y_true_bin, y_pred_bin, zero_division=0)
        }

    # Export the rules to a CSV file
    def export_rules(self, filename: str):
        """
        Export the generated rules to a CSV file.
        Parameters:
        - filename: The name of the CSV file to save the rules.
        """
        self.rules_df.to_csv(filename, index=False)

    # Calculate Jaccard similarity between rules generated by different algorithms
    def jaccard_similarity(self, rules_a: pd.DataFrame, rules_b: pd.DataFrame) -> float:
        """
        Calculate the Jaccard similarity between the rule sets of two algorithms.
        The Jaccard similarity measures the overlap between two sets of rules.
        """
        set_a = set(zip(rules_a['antecedents'], rules_a['consequents']))
        set_b = set(zip(rules_b['antecedents'], rules_b['consequents']))
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return round(len(intersection) / len(union), 4) if union else 0.0

    # Prepare the rules for web output, format antecedents and consequents as strings
    def prepare_for_web(self) -> pd.DataFrame:
        """
        Prepare the rules DataFrame for web output by formatting antecedents and consequents as strings.
        """
        df = self.rules_df.copy()
        df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        df['consequents'] = df['consequents'].apply(lambda x: ', '.join(sorted(x)))
        return df[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'composite_score', 'algorithm']]
