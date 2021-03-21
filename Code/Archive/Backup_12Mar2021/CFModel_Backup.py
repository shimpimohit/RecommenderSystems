# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:26:34 2020

@author: mashimpi
"""

# ------ COLLABORTIVE FILTERING RECOMMENDER MODEL ---------- #
import pandas as pd
import numpy as np

class CFModel:
    
    # Defining some constants
    
    TOP_N = 1000 # Number of Recommendations to be generated
    KNN = 30 # K-Nearest-Neigbours
    
    def __init__(self, movies,ratings,CFMethod = 'User',EliminateBias = True):
        self.movies = movies
        self.ratings = ratings
        self.CFMethod = CFMethod
        self.EliminateBias = EliminateBias
        
    def GenerateUserMovieMatrix(self):
        # Normalize the Ratings by eliminating BIAS
        if (self.CFMethod.upper() == 'USER'):
            # Condition for USER-USER CF
            ratings_UserMean = self.ratings.groupby(['userId'],as_index=False,\
                                                    sort = False)\
                                           .mean()\
                                           .rename(columns={"rating":"MeanUserRating"})\
                                                [['userId','MeanUserRating']]
            self.ratings = pd.merge(self.ratings,ratings_UserMean,\
                                                 on='userId',how='left')
            self.ratings['AdjustedRating'] = self.ratings['rating'] \
                                              - self.ratings['MeanUserRating']
        else:
            # Condition for ITEM-ITEM CF
            ratings_MovieMean = self.ratings.groupby(['movieId'],as_index = False,sort = False)\
                                         .mean()\
                                         .rename(columns={"rating":"MeanMovieRating"})\
                                                [['movieId','MeanMovieRating']]
            self.ratings = pd.merge(self.ratings,ratings_MovieMean,\
                                    on='movieId',how='left')
            
            self.ratings['AdjustedRating'] = self.ratings['rating'] \
                                             - self.ratings['MeanMovieRating']
            
        # Generate User-Movie Matrix
        if self.EliminateBias == False:
            UserMovieMatrix = self.ratings.pivot_table(index='userId',\
                                                       columns='movieId',\
                                                       values='rating')\
                                  .fillna(0)
        else:
            UserMovieMatrix = self.ratings.pivot_table(index='userId',\
                                                       columns='movieId',\
                                                       values='AdjustedRating') \
                                          .fillna(0)
        
        return UserMovieMatrix

    def GenerateSimilarityMatrix(self,UserMovieMatrix):
        # Generate COSINE SIMILARITY MATRIX
        # Cos(A,B) = (A.B) / |A||B|
        # Add epsilon=1e-9 to the DOT Products so that we avoid DIVIDE BY ZERO
        
        if self.CFMethod.upper() == 'USER':
            # User-User Dot Product
            DotProduct = UserMovieMatrix.dot(UserMovieMatrix.T) + 1e-9
        else:
            # Item-Item Dot Product
            DotProduct = UserMovieMatrix.T.dot(UserMovieMatrix) + 1e-9
        
        Magnitude = np.array([np.sqrt(np.diag(DotProduct))])
        CosineSimilarityMatrix = DotProduct / Magnitude / Magnitude.T
                              
        return CosineSimilarityMatrix

    def PredictRating(self, UserMovieMatrix, SimilarityMatrix, k=KNN):
        PredictedRatings = pd.DataFrame(0, index=UserMovieMatrix.index, \
                                           columns=UserMovieMatrix.columns)
            
        if self.CFMethod.upper() == 'USER':
            # Condition for USER-USER CF
            for user in UserMovieMatrix.index:
                print('UserID: ',user)
                TopKSimilarUsers = SimilarityMatrix.loc[user]\
                                                   .sort_values(ascending=False)\
                                                   .index [1:1+k]
                for movie in UserMovieMatrix.columns:
                    PredictedRatings.loc[user][movie] = \
                        SimilarityMatrix.loc[TopKSimilarUsers][user]\
                                        .dot(UserMovieMatrix\
                                        .loc[TopKSimilarUsers][movie])
                    PredictedRatings.loc[user][movie] /= \
                        np.sum(np.abs(SimilarityMatrix\
                                      .loc[TopKSimilarUsers][user]))
        else:
            # Condition for ITEM-ITEM CF
            for movie in UserMovieMatrix.columns:
                print('MovieID: ',movie)
                TopKSimilarMovies = SimilarityMatrix.loc[movie]\
                                                   .sort_values(ascending=False)\
                                                   .index [1:1+k]
                for user in UserMovieMatrix.index:
                    PredictedRatings.loc[user][movie] = \
                        SimilarityMatrix.loc[TopKSimilarMovies][movie]\
                                        .dot(UserMovieMatrix\
                                        .loc[user][TopKSimilarMovies])
                    PredictedRatings.loc[user][movie] /= \
                        np.sum(np.abs(SimilarityMatrix\
                                      .loc[TopKSimilarMovies][movie]))
            
        return PredictedRatings
        
    def PredictUserRating(self, UserId, k=KNN):
        UserMovieMatrix = self.GenerateUserMovieMatrix()
        SimilarityMatrix = self.GenerateSimilarityMatrix(UserMovieMatrix)
        PredictedUserRatings = pd.DataFrame(0, index=pd.Series(UserId), \
                                           columns=UserMovieMatrix.columns)
            
        if self.CFMethod.upper() == 'USER':
            # Condition for USER-USER CF
            TopKSimilarUsers = SimilarityMatrix.loc[UserId]\
                                                   .sort_values(ascending=False)\
                                                   .index[1:1+k]
            for movie in UserMovieMatrix.columns:
                #print("Movie ID : ",movie)
                
                PredictedUserRatings[movie] = \
                    SimilarityMatrix.loc[TopKSimilarUsers][UserId]\
                        .dot(UserMovieMatrix.loc[TopKSimilarUsers][movie])
                    
                PredictedUserRatings[movie] /= \
                    np.sum(np.abs(SimilarityMatrix.loc[TopKSimilarUsers][UserId]))
        else:
            # Condition for ITEM-ITEM CF
            for movie in UserMovieMatrix.columns:
                TopKSimilarMovies = SimilarityMatrix.loc[movie]\
                                                   .sort_values(ascending=False)\
                                                   .index [1:1+k]
                
                PredictedUserRatings[movie] = \
                        SimilarityMatrix.loc[TopKSimilarMovies][movie]\
                                        .dot(UserMovieMatrix\
                                        .loc[UserId][TopKSimilarMovies])
                PredictedUserRatings[movie] /= \
                        np.sum(np.abs(SimilarityMatrix\
                                      .loc[TopKSimilarMovies][movie]))
                            
        return PredictedUserRatings
                    
    
    def GetModel(self):
        if (self.CFMethod.upper() == 'USER'):
            # Condition for USER-USER CF
            MODEL = 'User-User-CF'
        else:
            # Condition for ITEM-ITEM CF
            MODEL = 'Item-Item-CF'
        return MODEL
                                      
    # Get Already rated movies that need to be excluded from recommendations
    def GetUserRatedMovies(self, UserId):
        RatedMovies = self.ratings.set_index('userId').loc[UserId]['movieId']
        return RatedMovies

    def RecommendMovies(self,UserId):
        #Predict User Ratings
        PredictedUserRatings = self.PredictUserRating(UserId)
        
        # Exclusion List of all movies that user has already rated
        ExcludeList = self.GetUserRatedMovies(UserId)
        
        UserRecommendations = list(set(PredictedUserRatings.columns.tolist()) \
                                   - set(ExcludeList.tolist()))
            
        UserRecommendations = pd.DataFrame(UserRecommendations,columns=['movieId'])
        UserRecommendations = UserRecommendations.merge(PredictedUserRatings.T,\
                                                        how='left',\
                                                        left_on = 'movieId',\
                                                        right_on = 'movieId')
        UserRecommendations.columns = ['movieId','PredictedRating']
        UserRecommendations = UserRecommendations\
                                    .sort_values(by = ['PredictedRating']\
                                                 ,ascending = False)\
                                    .head(self.TOP_N)
        
        UserRecommendations = UserRecommendations.merge(self.movies,\
                                                        how = 'left', \
                                                        left_on = 'movieId',\
                                                        right_on = 'movieId')
        
        return UserRecommendations[['movieId','title','genres','PredictedRating']] 

    

    