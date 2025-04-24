import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
class CollaborativeFiltering:
    ### Put Your Code Here
    def __init__(self, dataset_path, split_to_dev = False):
        if not dataset_path:
            return "Please provide path to the dataset file"
        dataset =  pd.read_csv(dataset_path)
        dataset = dataset.dropna(how='all')
        dataset = dataset.rename(columns={"User_id": "user_id", "Date": "date","itemDescription": "item_description"})
        
        transaction_history = self.__generate_transaction_history(dataset)
        utility_matrix = self.__generate_utility_matrix(dataset)
        item_norm_utility_matrix = self.__generate_normalize_utility_matrix(utility_matrix)
        item_similarity_matrix = self.__generate_item_similarity_matrix(item_norm_utility_matrix)
        
        self.transaction_history = transaction_history
        self.utility_matrix = utility_matrix
        self.item_norm_utility_matrix = item_norm_utility_matrix
        self.item_similarity_matrix = item_similarity_matrix
        

    def __generate_transaction_history(self, dataset):
        transaction_history = dataset.groupby(['user_id', 'date', "year", "month", "day", "day_of_week"]).agg({
            'item_description': list
        }).reset_index()

        transaction_history = transaction_history.rename(columns={"item_description": "items"})
        return transaction_history

    def __generate_utility_matrix(self, dataset):
        utility_matrix = pd.crosstab(dataset['user_id'].astype(int), dataset['item_description'])
        utility_matrix.replace(0, np.nan, inplace=True)
        return utility_matrix
        
      
    def __mean_norm(self, array):
        mean = np.mean(array)
        norm_array = [freq - mean for freq in array]
        return pd.Series(norm_array, index=array.index), mean    

    def __generate_normalize_utility_matrix(self, utility_matrix, type="item"):
        
        axis = 0 if type == "item" else 1
        mean_dict = {}
        
        def apply_mean_norm(x):
            norm_x, mean = self.__mean_norm(x)
            mean_dict[x.name] = mean 
            return norm_x
            
        utility_matrix_normalized = utility_matrix.apply(apply_mean_norm, axis=axis)
        return utility_matrix_normalized
        
    def __generate_item_similarity_matrix(self, norm_utility_matrix):
        filled_matrix = norm_utility_matrix.fillna(0).T  # Transpose: items as rows
    
        similarity = cosine_similarity(filled_matrix)
     
        similarity_dataframe = pd.DataFrame(similarity, 
                                    index=filled_matrix.index, 
                                    columns=filled_matrix.index)

        return similarity_dataframe
  

    def __predict_rating(self, 
                         item_similarity_matrix, 
                         utility_matrix, 
                         user_purchased_items, 
                         target_user, 
                         target_item, 
                         n_neighbour):
        
        
        similar_items = item_similarity_matrix[target_item].drop(index=target_item)
        top_k_items = similar_items.sort_values(ascending=False).head(n_neighbour)
        
        weighted_sum = 0
        similarity_sum = 0
        
        if len(user_purchased_items) < 1:
            return  utility_matrix[target_item].mean() 
        
        for item, similarity in top_k_items.items():
            if item in user_purchased_items:
                rating = utility_matrix.loc[target_user, item]
                if not np.isnan(rating):
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
    
        return weighted_sum / similarity_sum if similarity_sum != 0 else 0
    
    def __extract_items_from_frozenset(self, frozenset_str):
        """Extract item names from frozenset string representation."""
        # Example: frozenset({'soda'}) -> ['soda']
        # Remove "frozenset(" from start and ")" from end
        items_str = frozenset_str.replace("frozenset(", "").rstrip(")")
        
        # Handle either {'item'} or {'item1', 'item2'} formats
        items = []
        if items_str:
            # Strip outer braces
            content = items_str.strip("{}")
            # Split by comma and clean up quotes
            for item in content.split(","):
                cleaned_item = item.strip().strip("'\"")
                if cleaned_item:
                    items.append(cleaned_item)
        
        return items
    
    def __normalize_scores(self, recommendations):
        if not recommendations:
            return []
        max_score = max(score for _, score in recommendations)
        if max_score == 0:
            return [(item, 0.0) for item, _ in recommendations]
        return [(item, score / max_score) for item, score in recommendations]
    
    def __calculate_confidence_factor(self, k, t):
        alpha = min(t, k) / k * 0.9
        return alpha
    
    def __calculate_sigmoid(self, x, steepness=1.0, midpoint=2.0):
        return 1 / (1 + np.exp(-steepness * (x - midpoint)))
    
    def __get_cf_rule_weight(self, user_transaction_history,
                       n_recommended_items,
                       adaptive_switching_type="confidence_factor",
                       sigmoid_steepness=1.0,
                       sigmoid_midpoint=2.0,
                       ):
    
        history_length = len(user_transaction_history)
        if adaptive_switching_type == "sigmoid":
            user_history_weight = self.__calculate_sigmoid(history_length, 
                                        steepness=sigmoid_steepness, 
                                        midpoint=sigmoid_midpoint)
            cf_weight = 0.7 * user_history_weight
            rule_weight = 0.3 + (0.5 * (1 - user_history_weight))
            return cf_weight, rule_weight

        user_history_weight = self.__calculate_confidence_factor(n_recommended_items, history_length)
        cf_weight = user_history_weight
        rule_weight = 1 - user_history_weight
        
        return cf_weight, rule_weight

        
    def get_normalize_utility_matrix(self):
        return self.item_norm_utility_matrix
        
    def get_utility_matrix(self):
        return self.utility_matrix
    
    def get_item_similarity_matrix(self):
        return self.item_similarity_matrix
    
    def get_association_rules_recommendations(self, purchased_items, association_rules, max_recommendations=50):
        recommendations = {}
        purchased_set = set(purchased_items)
        if len(purchased_set) < 1:
            all_consequents = []
            for _, rule in association_rules.iterrows():
                all_consequents.extend(self.__extract_items_from_frozenset(rule['consequents']))

            most_common = Counter(all_consequents).most_common(max_recommendations)
            return [(item, 1) for item, _ in most_common]
            
        else:
            for _, rule in association_rules.iterrows():
                antecedent_items = self.__extract_items_from_frozenset(rule['antecedents'])
                consequent_items = self.__extract_items_from_frozenset(rule['consequents'])
                
                # Check if any purchased items are in the antecedent
                if any(item in purchased_set for item in antecedent_items):
                    for consequent_item in consequent_items:
                        # Skip if user already has this item
                        if consequent_item in purchased_set:
                            continue
                        rule_score = rule['composite_score_with_recency']
                        
                        # Update recommendations
                        recommendations[consequent_item] = max(
                            recommendations.get(consequent_item, 0),
                            rule_score
                        )
            
            # Sort recommendations by score
            sorted_recommendations = sorted(
                recommendations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_recommendations]
            
            return sorted_recommendations
    
    def get_cf_recommended_items(self, target_user, 
                                 n_recommended_items=None, 
                                 n_similar_neighbours = 10):
        
        all_user_ids = self.utility_matrix.index   
        user_unpurchased_items = []
        user_purchased_items = []
        if not target_user in all_user_ids:
            user_unpurchased_items = self.utility_matrix.columns 
        else:
            user_ratings_row = self.utility_matrix.loc[target_user]
            user_unpurchased_items = user_ratings_row[user_ratings_row.isna()].index.tolist()
            user_purchased_items = user_ratings_row[~user_ratings_row.isna()].index.tolist()
            
        rating_predictions = {}
        for target_item in user_unpurchased_items:
            predicted_rating = self.__predict_rating(self.item_similarity_matrix, 
                                                     self.utility_matrix, 
                                                     user_purchased_items, 
                                                     target_user, 
                                                     target_item,
                                                     n_similar_neighbours,
                                                     )
            rating_predictions[target_item] = predicted_rating
        
        recommended_items = sorted(rating_predictions.items(), key=lambda x: x[1], reverse=True)
        if not n_recommended_items:
            return recommended_items
        return recommended_items[:n_recommended_items]
    
    
    
    def weighted_hybrid_cf_recommended_items(self, association_rules, 
                                             cf_recommendations, 
                                             target_user,
                                             adaptive_switching_type="confidence_score"
                                             ):
        user_purchased_items = []
        user_transaction_history = []
        if target_user in self.utility_matrix.index.values: ## If user has transaction history
            user_ratings_row = self.utility_matrix.loc[target_user]
            user_purchased_items = user_ratings_row[~user_ratings_row.isna()].index.tolist()
            user_transaction_history = self.transaction_history[self.transaction_history["user_id"] == target_user]
    
        k = len(cf_recommendations)    
        cf_weight, rule_weight = self.__get_cf_rule_weight(user_transaction_history, k, adaptive_switching_type=adaptive_switching_type)
        association_rules_recommendations = self.get_association_rules_recommendations(user_purchased_items, 
                                                                                  association_rules, 
                                                                                  max_recommendations=k)
        final_scores = {}
        
        # Normalize Both Recommendations
        cf_recs_norm = self.__normalize_scores(cf_recommendations)
        ar_recs_norm = self.__normalize_scores(association_rules_recommendations)
    
        for item, score in cf_recs_norm:
            final_scores[item] = score * cf_weight
            
        for item, score in ar_recs_norm:
            if item in final_scores:
                final_scores[item] += score * rule_weight
            else:
                final_scores[item] = score * rule_weight
        
        # Sort and return top recommendations
        top_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:]
        return top_recommendations[:k]
            