# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:54:17 2021

@author: mashimpi
"""

# ----------- MODEL BASED COLLABORTIVE FILTERING RECOMMENDERS --------------- #

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class MFModel:
    # In this class we will explore Matrix Factorization models like SVD
    # Note: 
    #    a. For the purpose of this project, I am not writing the iterative 
    #       code to factorize the Rating Matrix into orthogonal factors
    #    b. I will be using the out-of-box models available in SciPy / scikit-learn
    
    
    # Defining Global Constants
    TOP_N = 1000 # Number of Recommendations to be generated
    FACTORS = 15 # This specifies the number of Factors for Matrix Factorization
    
    def __init__(self, movieData ,model = 'SVD'):
        self.movieData = movieData
        self.model = model
        self.movies = self.movieData.movies
        self.ratings = self.movieData.ratings
        self.UserMovieMatrixTrain = self.movieData.UserMovieMatrixTrain.fillna(0)
        self.UserMovieMatrixTest = self.movieData.UserMovieMatrixTest.fillna(0)
        self.PredictedRatingsTrain, self.PredictedRatingsTest = self.PredictRatings()
        
    
    #def GenerateUserMovieMatrix(self):
    #    UserMovieMatrix = self.ratings.pivot_table(index='userId',\
    #                                               columns='movieId',\
    #                                               values='rating').fillna(0)
    #    return UserMovieMatrix     
    
    def GenerateSparseUserMovieMatrix(self,UserMovieMatrix):
        return csr_matrix(UserMovieMatrix.to_numpy())
    
    def NormalizeRatings(self,UserRatingMatrix):
        NormalizedRatings = UserRatingMatrix - UserRatingMatrix.min() / \
                              (UserRatingMatrix.max() - UserRatingMatrix.min())
        return NormalizedRatings
    
    def FactorizeUserMovieMatrix(self,UserMovieMatrix):
        # This is the function that factorizes the User-Movie Matrix
        # based on the model - SVD, PMF, etc.
        
        if self.model.upper() == 'SVD':
            U, SIGMA, V = svds(UserMovieMatrix, k = self.FACTORS)
            SIGMA = np.diag(SIGMA)
        #else:
            # This is the CUSTOM MATRIX FACTORIZATION code
            
        return (U, SIGMA, V)
        
    def PredictRatings(self):
        
        SparseUserMovieMatrixTrain = self.GenerateSparseUserMovieMatrix(self.UserMovieMatrixTrain)
        SparseUserMovieMatrixTest = self.GenerateSparseUserMovieMatrix(self.UserMovieMatrixTest)
        SparseUserMovieMatrixTrain = self.NormalizeRatings(SparseUserMovieMatrixTrain)
        SparseUserMovieMatrixTest = self.NormalizeRatings(SparseUserMovieMatrixTest)
        
        UTrain, SIGMATrain, VTrain = self.FactorizeUserMovieMatrix(SparseUserMovieMatrixTrain)
        UTest, SIGMATest, VTest = self.FactorizeUserMovieMatrix(SparseUserMovieMatrixTest)
        PredictedRatingsTrain = np.dot(np.dot(UTrain,SIGMATrain),VTrain)
        PredictedRatingsTest = np.dot(np.dot(UTest,SIGMATest),VTest)
        
        PredictedRatingsTrain = pd.DataFrame(PredictedRatingsTrain, \
                                        index = self.UserMovieMatrixTrain.index,\
                                        columns = self.UserMovieMatrixTrain.columns)
        PredictedRatingsTest = pd.DataFrame(PredictedRatingsTest, \
                                        index = self.UserMovieMatrixTest.index,\
                                        columns = self.UserMovieMatrixTest.columns)
        return PredictedRatingsTrain,PredictedRatingsTest
    
    def GetModel(self):
        if (self.model.upper() == 'SVD'):
            MODEL = 'Scipy-SVD'
        
        return MODEL
                                      
    # Get Already rated movies that need to be excluded from recommendations
    def GetUserRatedMovies(self, UserId,ratings):
        RatedMovies = self.ratings.set_index('userId').loc[UserId]['movieId']
        return RatedMovies
    
    def RecommendMovies(self,UserId,RecommendForTest = True):
        #Predict User Ratings
        if RecommendForTest:
            PredictedRatings = self.PredictedRatingsTest
            ratings = self.movieData.ratingsTest
        else:
            PredictedRatings = self.PredictedRatingsTrain
            ratings = self.movieData.ratingsTrain
            
        PredictedUserRatings = PredictedRatings.loc[UserId]
        
        # Exclusion List of all movies that user has already rated
        ExcludeList = self.GetUserRatedMovies(UserId,ratings)
        
        UserRecommendations = list(set(PredictedUserRatings.index.tolist()) \
                                   - set(ExcludeList.tolist()))
            
        UserRecommendations = pd.DataFrame(UserRecommendations,columns=['movieId'])
        UserRecommendations = UserRecommendations.merge(PredictedUserRatings,\
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
    
    
    
    
    
    
    
        
        
    
    


