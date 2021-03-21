# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:26:34 2020

@author: mashimpi
"""

# ------ COLLABORTIVE FILTERING RECOMMENDER MODEL ---------- #
import pandas as pd
import numpy as np

class CFModel:
    
    # Defining some constants that will be shared by all instances of the class
    
    TOP_N = 1000 # Number of Recommendations to be generated
    
    # Constructor / initialization of CF Model
    # Parameters:
    #       1. movie (movie dataframe)
    #       2. ratings (this is the training data for user ratings)
    #       3. CF Method (this defines whether USER-USER or ITEM-ITEM)
    #       3. k - For indicating the number of nearst neighbours
    #       4. Similarity Metric (this defines the similarity methodology)
    #       5. Whether user or item bias has to be considered in prediction
    
    def __init__(self, 
                 movies,ratings,
                 CFMethod = 'User',
                 k = 0,
                 Metric = 'Cosine',
                 EliminateBias = True ):
        
        self.movies = movies
        self.ratings = ratings
        self.CFMethod = CFMethod.upper()
        self.k = k
        self.Metric = Metric.upper()
        self.EliminateBias = EliminateBias
        self.UserMovieMatrix = self.GenerateUserMovieMatrix()
        
        # Exceptional Handling for k
        #if self.CFMethod == 'USER':
        #    if self.k == 0 or self.k > len(self.UserMovieMatrix.index.unique()):
        #        self.k = len(self.UserMovieMatrix.index.unique())
        #else:
        #    if (k == 0 or k > len(self.UserMovieMatrix.columns.unique())):
        #        self.k = len(self.UserMovieMatrix.columns.unique())
        
        self.UserMeanRating = pd.DataFrame(np.nanmean(self.UserMovieMatrix,\
                                                      axis=1),\
                                              index=self.UserMovieMatrix.index)
        self.ItemMeanRating = pd.DataFrame(np.nanmean(self.UserMovieMatrix,\
                                                      axis=0),\
                                              index = self.UserMovieMatrix.columns)
        self.SimilarityMatrix = self.GenerateSimilarityMatrix()
        
        if self.EliminateBias == True:
            # Remove User Bias by using the formula
            #        RatingNoBias = Rating - MEAN USER RATING
            if self.CFMethod == 'USER':
                self.UserMovieMatrix = self.UserMovieMatrix\
                                           .sub(self.UserMeanRating.values)\
                                           .fillna(0)
            else:
                self.UserMovieMatrix = self.UserMovieMatrix\
                                           .sub(self.ItemMeanRating.T.values)\
                                           .fillna(0)
        else:
            self.UserMovieMatrix = self.UserMovieMatrix.fillna(0)
        self.PredictedRatings = self.PredictRating()        
        
    def GenerateUserMovieMatrix(self):
        # Simple Pivot Table of User-Movies
        UserMovieMatrix = self.ratings.pivot_table(index='userId',\
                                                       columns='movieId',\
                                                       values='rating')
        return UserMovieMatrix

    def GenerateSimilarityMatrix(self):
        # Generate SIMILARITY MATRIX
        #      a. COSINE SIMILARITY Cos(A,B) = (A.B) / |A|*|B|
        #      b. PEARSON SIMILARITY PSim(A,B) = (A-Amean).(B-Bmean) / 
        #                                             |A-Amean|*|B-Bmean|
        # Add epsilon=1e-9 to the DOT Products so that we avoid DIVIDE BY ZERO
        
        if self.CFMethod == 'USER':
            # User-User Dot Product
            if self.Metric == 'COSINE':
                # Define A = User-Movie Matrix 
                #    and B = Movie-User Matrix (or Transpose of A)
                #  So, A.B = User-User Matrix
                A = (self.UserMovieMatrix.fillna(0))
                B = A.T 
                DotProduct = A.dot(B) + 1e-9
                
            elif self.Metric == 'PEARSON':
                A = ((self.UserMovieMatrix.sub(np.nanmean(self.UserMovieMatrix,axis=1),\
                                          axis=0))\
                                     .fillna(0))
                B = A.T
                DotProduct = A.Dot(B) + 1e-9
            else:
                # Default to Cosine Similarity
                A = (self.UserMovieMatrix.fillna(0))
                B = A.T 
                DotProduct = A.dot(B) + 1e-9
        else:
            # Item-Item Dot Product
            
            if self.Metric == 'COSINE':
                # Define A = Movie-User Matrix (or Transpose of User-Movie)
                #    and B = User-Movie Matrix 
                #  So, A.B = Movie-Movie Matrix
                B = (self.UserMovieMatrix.fillna(0))
                A = B.T
                DotProduct = A.dot(B) + 1e-9
            elif self.Metric == 'PEARSON':
                A = self.UserMovieMatrix.T.sub(np.nanmean(self.UserMovieMatrix.T,\
                                                     axis=1),\
                                          axis=0).fillna(0)
                B = A.T
                DotProduct = A.dot(B) + 1e-9
            else:
                # Default to Cosine Similarity
                B = (self.UserMovieMatrix.fillna(0))
                A = B.T
                DotProduct = A.dot(B) + 1e-9
        
        
        Magnitude = np.array([np.sqrt(np.diag(DotProduct))])
        SimilarityMatrix = DotProduct / Magnitude / Magnitude.T
                              
        return SimilarityMatrix

    def PredictRating(self):
        
        PredictedRatings = pd.DataFrame(0, index=self.UserMovieMatrix.index, \
                                           columns=self.UserMovieMatrix.columns)
            
        if self.EliminateBias == True:
            if self.CFMethod == 'USER':
                # Default the Value of Predicted Rating to Mean User Rating
                # We will then add the dot product of Similarity and Adj Rating
                PredictedRatings = PredictedRatings.add(self.UserMeanRating.to_numpy())
            else:
                # Default the Value of Predicted Rating to Mean Item Rating
                # We will then add the dot product of Similarity and Adj Rating
                PredictedRatings = PredictedRatings.add(self.ItemMeanRating.T.to_numpy())
        #print(PredictedRatings.head(20))
        if self.CFMethod == 'USER':
            # Condition for USER-USER CF
            if self.k == 0 or self.k > len(self.UserMovieMatrix.index):
                PredictedRatings += np.dot(self.SimilarityMatrix.to_numpy(),\
                                          self.UserMovieMatrix.to_numpy()) \
                                   / np.sum(np.abs(self.SimilarityMatrix.to_numpy()))
            else:
                for user in self.UserMovieMatrix.index.to_numpy():
                    
                    TopKSimilarUsers = self.SimilarityMatrix.loc[user]\
                                           .sort_values(ascending=False)[1:self.k+1]\
                                           .index.to_numpy()
                    TopKSimilarityScore = self.SimilarityMatrix.loc[user]\
                                              .sort_values(ascending=False)\
                                                  [1:self.k+1].to_numpy()
                    
                    TopKSimilarityScoreNorm = np.sum(np.abs(TopKSimilarityScore))
                    PredictedRatings.loc[user] += np.dot(TopKSimilarityScore,\
                                                         self.UserMovieMatrix\
                                                             .loc[TopKSimilarUsers]) \
                                                 / TopKSimilarityScoreNorm
        else:
            # Condition for ITEM-ITEM CF
            PredictedRatings = PredictedRatings.T
            if self.k == 0 or self.k > len(self.UserMovieMatrix.columns):
                PredictedRatings += np.dot(self.SimilarityMatrix.to_numpy(),\
                                          self.UserMovieMatrix.to_numpy().T) \
                                   / np.sum(np.abs(self.SimilarityMatrix.to_numpy()))
            else:
                for movie in self.UserMovieMatrix.columns.to_numpy():
                    #print('MovieID: ',movie)
                    TopKSimilarMovies = self.SimilarityMatrix.loc[movie]\
                                            .sort_values(ascending=False)[1:1+self.k]\
                                            .index.to_numpy()
                    #print(TopKSimilarMovies)
                    TopKSimilarityScore = self.SimilarityMatrix.loc[movie]\
                                            .sort_values(ascending=False)[1:1+self.k]\
                                            .to_numpy()
                    #print(TopKSimilarityScore)
                    TopKSimilarityScoreNorm = np.sum(np.abs(TopKSimilarityScore))
                    #print(TopKSimilarityScoreNorm)
                    PredictedRatings.loc[movie] += np.dot(TopKSimilarityScore, \
                                                          self.UserMovieMatrix.T\
                                                              .loc[TopKSimilarMovies])\
                                                   / TopKSimilarityScoreNorm
            PredictedRatings = PredictedRatings.T
        return PredictedRatings            

    
    # def PredictUserRating(self, UserId, k=KNN):
    #     PredictedUserRatings = pd.DataFrame(0, index=pd.Series(UserId), \
    #                                        columns=self.UserMovieMatrix.columns)
        
    #     if self.EliminateBias == True:
    #         if self.CFMethod.upper() == 'USER':
    #             # Default the Value of Predicted Rating to Mean User Rating
    #             # We will then add the dot product of Similarity and Adj Rating
    #             PredictedUserRatings = PredictedUserRatings\
    #                                         .add(self.UserMeanRating.loc[UserId].values,\
    #                                              axis=0)
    #         else:
    #             # Default the Value of Predicted Rating to Mean Item Rating
    #             # We will then add the dot product of Similarity and Adj Rating
    #             PredictedUserRatings = PredictedUserRatings.add(self.ItemMeanRating.T\
    #                                                             .values)
        
    #     if self.CFMethod.upper() == 'USER':
    #         # Condition for USER-USER CF
    #         TopKSimilarUsers = self.SimilarityMatrix.loc[UserId]\
    #                                 .sort_values(ascending=False)[1:k+1]\
    #                                 .index.to_numpy()
            
    #         TopKSimilarityScore = self.SimilarityMatrix.loc[UserId]\
    #                                 .sort_values(ascending=False)[1:k+1].to_numpy()
                
    #         TopKSimilarityScoreNorm = np.sum(np.abs(TopKSimilarityScore))
            
    #         PredictedUserRatings += np.dot(TopKSimilarityScore,\
    #                                        self.UserMovieMatrix.loc[TopKSimilarUsers]) \
    #                                          / TopKSimilarityScoreNorm
    #     else:
    #         # Condition for ITEM-ITEM CF
    #         for movie in self.UserMovieMatrix.columns:
    #             TopKSimilarMovies = self.SimilarityMatrix.loc[movie]\
    #                                                .sort_values(ascending=False)\
    #                                                .index [1:1+k]
                
    #             PredictedUserRatings[movie] += \
    #                     self.SimilarityMatrix.loc[TopKSimilarMovies][movie]\
    #                                     .dot(self.UserMovieMatrix\
    #                                     .loc[UserId][TopKSimilarMovies]) \
    #                         / (np.sum(np.abs(self.SimilarityMatrix\
    #                                              .loc[TopKSimilarMovies][movie])))

    #     return PredictedUserRatings
    
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
        PredictedUserRatings = self.PredictedRatings.loc[UserId]
        
        # Exclusion List of all movies that user has already rated
        ExcludeList = self.GetUserRatedMovies(UserId)
        
        UserRecommendations = list(set(PredictedUserRatings.index.tolist()) \
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

    

    