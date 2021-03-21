# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:03:00 2020

@author: mashimpi
"""

# Important Libraries
import pandas as pd
#import numpy as np

class UserItemStats:
    
    userRatingsPath = '../Data/ml-latest-small/ratings.csv'
    itemsPath = '../Data/ml-latest-small/movies.csv'
    tagsPath = '../Data/ml-latest-small/tags.csv'
    
    # Load DataSets
    def LoadUserRatings(self):
        print("Loading Ratings DataSet ...")
        return pd.read_csv(self.userRatingsPath)
    
    def LoadItems(self):
        print("Loading Items DataSet ...")
        return pd.read_csv(self.itemsPath)
    
    def LoadTags(self):
        print("Loading Tags DataSet ...")
        return pd.read_csv(self.tagsPath)
    
    def LoadDataSets(self):
        ratings = self.LoadUserRatings()
        print("Loaded",ratings.shape[0],"user ratings ...\n")
        
        items = self.LoadItems()
        print("Loaded",items.shape[0],"movies ...\n")
        
        tags = self.LoadTags()
        print("Loaded",tags.shape[0],"tags ...\n")
        
        print("Enriching Ratings Dataset with Movie Information ...\n")
        data = ratings.merge(items,on='movieId',how='left')
        print("Datasets Loaded.")
        return (items, ratings, tags, data)
    
    def GenerateItemStats(self,data):
        print("Generating Movie Statistics ...")
        
        # For Weighted Ratings, we define the PERCENTILE CUT-OFF in Vote Count   
        # For sake of simplicity, this is hardcoded rather than parameterised
        QUANTILE_THRESHOLD = 0.85
                
        #movie_stats_df = pd.DataFrame()
        item_stats_df = data[['movieId','title']].drop_duplicates()
        item_stats_df.set_index(['movieId'],inplace = True)
        item_stats_df['Rating_Mean'] = data.groupby('movieId')['rating'].mean()
        item_stats_df['Rating_Count'] = data.groupby('movieId')['rating'].count()
        
        # Lets Calculate Weighted Ratings
        # Weighted_Rating = (v / (v + m)) * R + (m / (v + m)) * C
        # where,
        #       v = Rating_Count or number of ratings each movie recieved
        #       m = minimum number of ratings required to qualify as popular
        #       R = Rating_Mean or Mean Rating for the movie
        #       C = Mean_Rating for all movies in the dataset
        
        C = data.rating.mean()
        m = item_stats_df['Rating_Count'].quantile(QUANTILE_THRESHOLD)
        
        item_stats_df['Weighted_Rating'] = (item_stats_df['Rating_Mean'] * 
                                             item_stats_df['Rating_Count'] / 
                                             (m + item_stats_df['Rating_Count'])) \
                                           + (m * C / (m + item_stats_df['Rating_Count']))
                                             
        return item_stats_df
    
    #General Funtions
    def getUserRatings(self, userID):
        return 1

MovieStats = UserItemStats()
movies, ratings, tags, data = MovieStats.LoadDataSets()
movie_stats_df = MovieStats.GenerateItemStats(data)

print('Top 10 Charts ...')
print(movie_stats_df.sort_values('Weighted_Rating', ascending=False).head(10))







