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
        # Simple Pivot Table of User-Movies
        UserMovieMatrix = self.ratings.pivot_table(index='userId',\
                                                       columns='movieId',\
                                                       values='rating')
                                  #.fillna(0)
        return UserMovieMatrix

    def GenerateSimilarityMatrix(self,UserMovieMatrix,Metric='Cosine'):
        # Generate SIMILARITY MATRIX
        #      a. COSINE SIMILARITY Cos(A,B) = (A.B) / |A|*|B|
        #      b. PEARSON SIMILARITY PSim(A,B) = (A-Amean).(B-Bmean) / 
        #                                             |A-Amean|*|B-Bmean|
        # Add epsilon=1e-9 to the DOT Products so that we avoid DIVIDE BY ZERO
        
        if self.CFMethod.upper() == 'USER':
            # User-User Dot Product
            if Metric.upper() == 'COSINE':
                # Define A = User-Movie Matrix 
                #    and B = Movie-User Matrix (or Transpose of A)
                #  So, A.B = User-User Matrix
                A = (UserMovieMatrix.fillna(0))
                B = A.T 
                DotProduct = A.dot(B) + 1e-9
                
            elif Metric.upper() == 'PEARSON':
                A = ((UserMovieMatrix.sub(np.nanmean(UserMovieMatrix,axis=1),\
                                          axis=0))\
                                     .fillna(0))
                B = A.T
                DotProduct = A.Dot(B)
            else:
                # Default to Cosine Similarity
                A = (UserMovieMatrix.fillna(0))
                B = A.T 
                DotProduct = A.dot(B) + 1e-9
        else:
            # Item-Item Dot Product
            if Metric.upper() == 'COSINE':
                # Define A = Movie-User Matrix (or Transpose of User-Movie)
                #    and B = User-Movie Matrix 
                #  So, A.B = Movie-Movie Matrix
                B = (UserMovieMatrix.fillna(0))
                A = B.T
                DotProduct = A.dot(B) + 1e-9
            elif Metric.upper() == 'PEARSON':
                A = UserMovieMatrix.T.sub(np.nanmean(UserMovieMatrix.T,\
                                                     axis=1),\
                                          axis=0).fillna(0)
                B = A.T
                DotProduct = A.dot(B) + 1e-9
            else:
                # Default to Cosine Similarity
                B = (UserMovieMatrix.fillna(0))
                A = B.T
                DotProduct = A.dot(B) + 1e-9
        
        Magnitude = np.array([np.sqrt(np.diag(DotProduct))])
        SimilarityMatrix = DotProduct / Magnitude / Magnitude.T
                              
        return SimilarityMatrix

    def PredictRating(self, k=KNN, SimilarityMetric='Cosine'):
        #Generate UserMovieMatrix and Similarity Matrix
        UserMovieMatrix = self.GenerateUserMovieMatrix()
        SimilarityMatrix = self.GenerateSimilarityMatrix(UserMovieMatrix,\
                                                         SimilarityMetric)
        
        #Initialize the data frame for storing predicted values
        PredictedRatings = pd.DataFrame(0, index=UserMovieMatrix.index, \
                                           columns=UserMovieMatrix.columns)
        
        if self.EliminateBias == True:
            if self.CFMethod.upper() == 'USER':
                # Remove User Bias by using the formula
                #        RatingNoBias = Rating - MEAN USER RATING
                
                UserMeanVector = np.nanmean(UserMovieMatrix,axis=1)
                UserMovieMatrix = UserMovieMatrix.sub(UserMeanVector,axis=0)\
                                                 .fillna(0)
                #UserMovieMatrix = UserMovieMatrix.sub(np.nanmean(UserMovieMatrix,\
                #                                                 axis=1),\
                #                                      axis=0).fillna(0)
                
                # Default the Value of Predicted Rating to Mean User Rating
                # We will then add the dot product of Similarity and Adj Rating
                PredictedRatings = PredictedRatings.add(UserMeanVector,axis=0)
            else:
                ItemMeanVector = np.nanmean(UserMovieMatrix,axis=0)
                UserMovieMatrix = UserMovieMatrix.sub(ItemMeanVector,axis=1)\
                                                 .fillna(0)
                #UserMovieMatrix = UserMovieMatrix.T.sub(np.nanmean(UserMovieMatrix.T,\
                #                                                   axis=1),\
                #                                        axis=0).T.fillna(0)
                
                # Default the Value of Predicted Rating to Mean Item Rating
                # We will then add the dot product of Similarity and Adj Rating
                PredictedRatings = PredictedRatings.add(ItemMeanVector,axis=1)
        else:
            UserMovieMatrix = UserMovieMatrix.fillna(0)
            
            
        if self.CFMethod.upper() == 'USER':
            # Condition for USER-USER CF
            for user in UserMovieMatrix.index:
                print('UserID: ',user)
                TopKSimilarUsers = SimilarityMatrix.loc[user]\
                                                   .sort_values(ascending=False)\
                                                   .index [1:1+k]
                for movie in UserMovieMatrix.columns:
                    PredictedRatings.loc[user][movie] += \
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
                    PredictedRatings.loc[user][movie] += \
                        SimilarityMatrix.loc[TopKSimilarMovies][movie]\
                                        .dot(UserMovieMatrix\
                                        .loc[user][TopKSimilarMovies])
                    PredictedRatings.loc[user][movie] /= \
                        np.sum(np.abs(SimilarityMatrix\
                                      .loc[TopKSimilarMovies][movie]))
            
        return PredictedRatings
        
    def PredictUserRating(self, UserId, k=KNN, SimilarityMetric='Cosine'):
        UserMovieMatrix = self.GenerateUserMovieMatrix()
        print('Debug - Actual User Ratings = ', UserMovieMatrix.loc[UserId])
        SimilarityMatrix = self.GenerateSimilarityMatrix(UserMovieMatrix,\
                                                         SimilarityMetric)
        PredictedUserRatings = pd.DataFrame(0, index=pd.Series(UserId), \
                                           columns=UserMovieMatrix.columns)
        
        if self.EliminateBias == True:
            if self.CFMethod.upper() == 'USER':
                # Remove User Bias by using the formula
                #        RatingNoBias = Rating - MEAN USER RATING
                
                #UserMeanVector = np.nanmean(UserMovieMatrix,axis=1)
                UserMeanRating = pd.DataFrame(np.nanmean(UserMovieMatrix,axis=1),\
                                              index=UserMovieMatrix.index)
                print('Debug - UserMean = ', UserMeanRating.loc[UserId])
                UserMovieMatrix = UserMovieMatrix.sub(UserMeanRating.values,\
                                                      axis=1)\
                                                 .fillna(0)
                print('Debug - Actual Adjusted User Ratings = ', UserMovieMatrix.loc[UserId])
                # Default the Value of Predicted Rating to Mean Item Rating
                # We will then add the dot product of Similarity and Adj Rating
                
                print('Debug - Default Prediction = ', PredictedUserRatings)
                PredictedUserRatings = PredictedUserRatings\
                                            .add(UserMeanRating.loc[UserId].values,\
                                                 axis=0)
            else:
                ItemMeanRating = pd.DataFrame(np.nanmean(UserMovieMatrix,axis=0),\
                                              index = UserMovieMatrix.columns)
                print('Debug - ItemMean = ', ItemMeanRating.loc[UserId])
                UserMovieMatrix = UserMovieMatrix.sub(ItemMeanRating.T.values,\
                                                      axis=1)\
                                                 .fillna(0)
                # Default the Value of Predicted Rating to Mean Item Rating
                # We will then add the dot product of Similarity and Adj Rating
                PredictedUserRatings = PredictedUserRatings.add(ItemMeanRating.T.values)
        else:
            UserMovieMatrix = UserMovieMatrix.fillna(0)
        
        if self.CFMethod.upper() == 'USER':
            # Condition for USER-USER CF
            TopKSimilarUsers = SimilarityMatrix.loc[UserId]\
                                                   .sort_values(ascending=False)\
                                                   .index[1:1+k]
            print('Debug - TopKSimilar Users = ', TopKSimilarUsers)
            for movie in UserMovieMatrix.columns:
                #print("Movie ID : ",movie)
                
                PredictedUserRatings[movie] += \
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
                
                PredictedUserRatings[movie] += \
                        SimilarityMatrix.loc[TopKSimilarMovies][movie]\
                                        .dot(UserMovieMatrix\
                                        .loc[UserId][TopKSimilarMovies])
                PredictedUserRatings[movie] /= \
                        np.sum(np.abs(SimilarityMatrix\
                                      .loc[TopKSimilarMovies][movie]))
                            
        print('Debug - Predicted Rating = ', PredictedUserRatings)            
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

    

    