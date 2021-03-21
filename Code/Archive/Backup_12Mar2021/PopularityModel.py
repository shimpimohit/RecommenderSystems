# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:01:37 2020

@author: mashimpi
"""


import pandas as pd

class PopularityModel:
    
    MODEL = 'Popularity'
    TOP_N = 5000 #Top N list of recommendations
    
    def __init__(self,movies,ratings,movieRankings):
        self.movies = movies
        self.ratings = ratings
        self.movieRankings = movieRankings
        
    def GetModel(self):
        return self.MODEL
    
    def GetUserRatedMovies(self, UserId):
        RatedMovies = self.ratings.set_index('userId').loc[UserId]['movieId']
        return RatedMovies
    
    def RecommendMovies(self, UserId):
        
        # Exclusion List of all movies that user has already rated
        ExcludeList = self.GetUserRatedMovies(UserId)
        
        UserRecommendations = list(set(self.movieRankings.index.tolist()) \
                                 - set(ExcludeList.tolist()))
        
        UserRecommendations = pd.DataFrame(UserRecommendations,columns=['movieId'])
        
        UserRecommendations = UserRecommendations.merge(self.movieRankings,\
                                                        how='left',\
                                                        left_on = 'movieId',\
                                                        right_on = 'movieId')
        
        UserRecommendations = UserRecommendations.merge(self.movies[['movieId','genres']],\
                                                        how='left',\
                                                        left_on = 'movieId',\
                                                        right_on = 'movieId')
        
        UserRecommendations.rename(columns={'Weighted_Rating':'PredictedRating'},\
                                   inplace=True)
        UserRecommendations = UserRecommendations.sort_values(\
                                                    by='PredictedRating',\
                                                    ascending=False)\
                                                 .head(self.TOP_N)
        
        return UserRecommendations[['movieId','title','genres','PredictedRating']]
    

        
