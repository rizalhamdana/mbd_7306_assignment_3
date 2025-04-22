import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    ### Put Your Code Here
    def __init__(self, dataset_path):
        if not dataset_path:
            return "Please provide path to the dataset file"
        dataset =  pd.read_csv(dataset_path)
        dataset = dataset.dropna(how='all')
        dataset = dataset.rename(columns={"User_id": "user_id", "Date": "date","itemDescription": "item_description"})
        utility_matrix = self.__generate_utility_matrix(dataset)
        item_norm_utility_matrix = self.__generate_normalize_utility_matrix(utility_matrix)
        item_similarity_matrix = self.__generate_item_similarity_matrix(item_norm_utility_matrix)
        
        self.utility_matrix = utility_matrix
        self.item_norm_utility_matrix = item_norm_utility_matrix
        self.item_similarity_matrix = item_similarity_matrix
        

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
  

    def __predict_rating(self, item_similarity_matrix, utility_matrix, user_purchased_items, target_user, target_item, n_neighbour):
        similar_items = item_similarity_matrix[target_item].drop(index=target_item)
        top_k_items = similar_items.sort_values(ascending=False).head(n_neighbour)
        
        weighted_sum = 0
        similarity_sum = 0
            
        for item, similarity in top_k_items.items():
            if item in user_purchased_items:
                rating = utility_matrix.loc[target_user, item]
                if not np.isnan(rating):
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
                
        if similarity_sum == 0:
            return 0.0
    
        return weighted_sum / similarity_sum
        
        
    def get_normalize_utility_matrix(self):
        return self.item_norm_utility_matrix
        
    def get_utility_matrix(self):
        return self.utility_matrix
    
    def get_item_similarity_matrix(self):
        return self.item_similarity_matrix
    
    def recommend_items(self, target_user, n_recommended_items=None, n_similar_neighbours = 10):
        all_user_ids = self.utility_matrix.index.unique()
        
        if not target_user in all_user_ids:
            return "Cold Start Problem Has Not Been Handled"
        
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