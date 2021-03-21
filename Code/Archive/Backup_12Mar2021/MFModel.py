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
    
    def __init__(self, movies,ratings,model = 'SVD'):
        self.movies = movies
        self.ratings = ratings
        self.model = model
    
    def GenerateUserMovieMatrix(self):
        UserMovieMatrix = self.ratings.pivot_table(index='userId',\
                                                   columns='movieId',\
                                                   values='rating').fillna(0)
        return UserMovieMatrix     
    
    def GenerateSparseUserMovieMatrix(self):
        UserMovieMatrix = self.ratings.pivot_table(index='userId',\
                                                   columns='movieId',\
                                                   values='rating').fillna(0)
        UserMovieMatrix = UserMovieMatrix.to_numpy()
        return csr_matrix(UserMovieMatrix)
    
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
        UserMovieMatrix = self.GenerateUserMovieMatrix()
        SparseUserMovieMatrix = self.GenerateSparseUserMovieMatrix()
        #UserMovieMatrix = self.NormalizeRatings(UserMovieMatrix)
        
        U, SIGMA, V = self.FactorizeUserMovieMatrix(SparseUserMovieMatrix)
        PredictedRatings = np.dot(np.dot(U,SIGMA),V)
        PredictedRatings = pd.DataFrame(PredictedRatings, \
                                        index = UserMovieMatrix.index,\
                                        columns = UserMovieMatrix.columns)
        return PredictedRatings
    
    def GetModel(self):
        if (self.model.upper() == 'SVD'):
            MODEL = 'Scipy-SVD'
        
        return MODEL
                                      
    # Get Already rated movies that need to be excluded from recommendations
    def GetUserRatedMovies(self, UserId):
        RatedMovies = self.ratings.set_index('userId').loc[UserId]['movieId']
        return RatedMovies
    
    def RecommendMovies(self,UserId):
        #Predict User Ratings
        PredictedRatings = self.PredictRatings()
        PredictedUserRatings = PredictedRatings.loc[UserId]
        
        # Exclusion List of all movies that user has already rated
        ExcludeList = self.GetUserRatedMovies(UserId)
        
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
    
    
    
    
    
    
    
        
        
    
    


