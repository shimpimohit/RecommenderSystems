# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:26:34 2020

@author: mashimpi
"""

# ------ COLLABORTIVE FILTERING RECOMMENDER MODEL ---------- #
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

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
                 movieData,
                 CFMethod = 'User',
                 k = 0,
                 Metric = 'Cosine',
                 EliminateBias = True,
                 modelMode = 'Fit'):
        print('Constructing Collaborative Model ...')
        self.movieData = movieData
        self.CFMethod = CFMethod.upper()
        self.k = k
        self.Metric = Metric.upper()
        self.EliminateBias = EliminateBias
        self.modelMode = modelMode.upper()
        print('Parameters -> Model = %s Based | Similarity Metric = %s | Cold Start = %r\n' \
              % (self.CFMethod,self.Metric, not self.EliminateBias))
        self.UserMovieMatrixTrain = self.movieData.UserMovieMatrixTrain
        self.UserMovieMatrixTest = self.movieData.UserMovieMatrixTest
        
        
        self.UserMeanRatingTrain = pd.DataFrame(np.nanmean(self.UserMovieMatrixTrain,\
                                                           axis=1),\
                                                index = self.UserMovieMatrixTrain\
                                                          .index)
        self.UserMeanRatingTest = pd.DataFrame(np.nanmean(self.UserMovieMatrixTest,\
                                                           axis=1),\
                                               index = self.UserMovieMatrixTest\
                                                           .index)    
        
        self.ItemMeanRatingTrain = pd.DataFrame(np.nanmean(self.UserMovieMatrixTrain,\
                                                           axis=0),\
                                                index = self.UserMovieMatrixTrain\
                                                            .columns)
        self.ItemMeanRatingTest = pd.DataFrame(np.nanmean(self.UserMovieMatrixTest,\
                                                          axis=0),\
                                               index = self.UserMovieMatrixTest\
                                                           .columns)
            
        self.SimilarityMatrix = self.GenerateSimilarityMatrix()
        
        if self.EliminateBias == True:
            # Remove User Bias by using the formula
            #        RatingNoBias = Rating - MEAN USER RATING
            if self.CFMethod == 'USER':
                self.UserMovieMatrixTrain = self.UserMovieMatrixTrain\
                                           .sub(self.UserMeanRatingTrain.values)\
                                           .fillna(0)
                self.UserMovieMatrixTest = self.UserMovieMatrixTest\
                                           .sub(self.UserMeanRatingTest.values)\
                                           .fillna(0)
            else:
                self.UserMovieMatrixTrain = self.UserMovieMatrixTrain\
                                           .sub(self.ItemMeanRatingTrain.T.values)\
                                           .fillna(0)
                self.UserMovieMatrixTest = self.UserMovieMatrixTest\
                                           .sub(self.ItemMeanRatingTest.T.values)\
                                           .fillna(0)
        else:
            self.UserMovieMatrixTrain = self.UserMovieMatrixTrain.fillna(0)
            self.UserMovieMatrixTest = self.UserMovieMatrixTest.fillna(0)
        #self.PredictedRatingsTrain, self.PredictedRatingsTest = self.PredictRating()       
        self.TrainModel()
        
    # def GenerateUserMovieMatrix(self):
    #     # Simple Pivot Table of User-Movies
    #     UserMovieMatrix = self.ratings.pivot_table(index='userId',\
    #                                                    columns='movieId',\
    #                                                    values='rating')
    #     return UserMovieMatrix

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
                A = (self.UserMovieMatrixTrain.fillna(0))
                B = A.T 
                DotProduct = A.dot(B) + 1e-9
                
            elif self.Metric == 'PEARSON':
                A = ((self.UserMovieMatrixTrain\
                          .sub(np.nanmean(self.UserMovieMatrixTrain,axis=1),\
                                          axis=0))\
                          .fillna(0))
                B = A.T
                DotProduct = A.dot(B) + 1e-9
            else:
                # Default to Cosine Similarity
                A = (self.UserMovieMatrixTrain.fillna(0))
                B = A.T 
                DotProduct = A.dot(B) + 1e-9
        else:
            # Item-Item Dot Product
            
            if self.Metric == 'COSINE':
                # Define A = Movie-User Matrix (or Transpose of User-Movie)
                #    and B = User-Movie Matrix 
                #  So, A.B = Movie-Movie Matrix
                B = (self.UserMovieMatrixTrain.fillna(0))
                A = B.T
                DotProduct = A.dot(B) + 1e-9
            elif self.Metric == 'PEARSON':
                A = self.UserMovieMatrixTrain.T\
                        .sub(np.nanmean(self.UserMovieMatrixTrain.T,\
                                        axis=1),\
                             axis=0).fillna(0)
                B = A.T
                DotProduct = A.dot(B) + 1e-9
            else:
                # Default to Cosine Similarity
                B = (self.UserMovieMatrixTrain.fillna(0))
                A = B.T
                DotProduct = A.dot(B) + 1e-9
        
        Magnitude = np.array([np.sqrt(np.diag(DotProduct))])
        SimilarityMatrix = DotProduct / Magnitude / Magnitude.T
                              
        return SimilarityMatrix

    def PredictRating(self):
        
        PredictedRatingsTrain = pd.DataFrame(0, index = self.UserMovieMatrixTrain\
                                                            .index, \
                                                columns=self.UserMovieMatrixTrain\
                                                            .columns)
        PredictedRatingsTest = pd.DataFrame(0, index = self.UserMovieMatrixTest\
                                                            .index, \
                                                columns=self.UserMovieMatrixTest\
                                                            .columns)    
        if self.EliminateBias == True:
            if self.CFMethod == 'USER':
                # Default the Value of Predicted Rating to Mean User Rating
                # We will then add the dot product of Similarity and Adj Rating
                PredictedRatingsTrain = PredictedRatingsTrain\
                                            .add(self.UserMeanRatingTrain\
                                                     .to_numpy())
                PredictedRatingsTest = PredictedRatingsTest\
                                            .add(self.UserMeanRatingTest\
                                                     .to_numpy())
            else:
                # Default the Value of Predicted Rating to Mean Item Rating
                # We will then add the dot product of Similarity and Adj Rating
                PredictedRatingsTrain = PredictedRatingsTrain\
                                            .add(self.ItemMeanRatingTrain.T\
                                                     .to_numpy())
                PredictedRatingsTest = PredictedRatingsTest\
                                            .add(self.ItemMeanRatingTest.T\
                                                     .to_numpy())
        
        if self.CFMethod == 'USER':
            # Condition for USER-USER CF
            if self.k == 0 or self.k > len(self.UserMovieMatrixTrain.index):
                PredictedRatingsTrain += np.dot(self.SimilarityMatrix.to_numpy(),\
                                                self.UserMovieMatrixTrain\
                                                    .to_numpy()) \
                                      / np.sum(np.abs(self.SimilarityMatrix\
                                                          .to_numpy()))
                PredictedRatingsTest += np.dot(self.SimilarityMatrix.to_numpy(),\
                                                self.UserMovieMatrixTest\
                                                    .to_numpy()) \
                                      / np.sum(np.abs(self.SimilarityMatrix\
                                                          .to_numpy()))
            else:
                for user in self.UserMovieMatrixTrain.index.to_numpy():
                    
                    TopKSimilarUsers = self.SimilarityMatrix.loc[user]\
                                           .sort_values(ascending=False)[1:self.k+1]\
                                           .index.to_numpy()
                    TopKSimilarityScore = self.SimilarityMatrix.loc[user]\
                                              .sort_values(ascending=False)\
                                                  [1:self.k+1].to_numpy()
                    
                    TopKSimilarityScoreNorm = np.sum(np.abs(TopKSimilarityScore))
                    PredictedRatingsTrain.loc[user] += np.dot(TopKSimilarityScore,\
                                                              self.UserMovieMatrixTrain\
                                                                  .loc[TopKSimilarUsers]) \
                                                     / TopKSimilarityScoreNorm
                    PredictedRatingsTest.loc[user] += np.dot(TopKSimilarityScore,\
                                                             self.UserMovieMatrixTest\
                                                                 .loc[TopKSimilarUsers]) \
                                                     / TopKSimilarityScoreNorm
        else:
            # Condition for ITEM-ITEM CF
            PredictedRatingsTrain = PredictedRatingsTrain.T
            PredictedRatingsTest = PredictedRatingsTest.T
            
            if self.k == 0 or self.k > len(self.UserMovieMatrixTrain.columns):
                PredictedRatingsTrain += np.dot(self.SimilarityMatrix.to_numpy(),\
                                                self.UserMovieMatrixTrain\
                                                    .to_numpy().T) \
                                      / np.sum(np.abs(self.SimilarityMatrix\
                                                          .to_numpy()))
                PredictedRatingsTest += np.dot(self.SimilarityMatrix.to_numpy(),\
                                               self.UserMovieMatrixTest\
                                                   .to_numpy().T) \
                                      / np.sum(np.abs(self.SimilarityMatrix\
                                                          .to_numpy()))
            else:
                for movie in self.UserMovieMatrixTrain.columns.to_numpy():
                    TopKSimilarMovies = self.SimilarityMatrix.loc[movie]\
                                            .sort_values(ascending=False)[1:1+self.k]\
                                            .index.to_numpy()
                    #print(TopKSimilarMovies)
                    TopKSimilarityScore = self.SimilarityMatrix.loc[movie]\
                                              .sort_values(ascending=False)\
                                                  [1:1+self.k]\
                                              .to_numpy()
                    #print(TopKSimilarityScore)
                    TopKSimilarityScoreNorm = np.sum(np.abs(TopKSimilarityScore))
                    #print(TopKSimilarityScoreNorm)
                    PredictedRatingsTrain.loc[movie] += \
                                        np.dot(TopKSimilarityScore, \
                                               self.UserMovieMatrixTrain.T\
                                                   .loc[TopKSimilarMovies])\
                                                     / TopKSimilarityScoreNorm
                    PredictedRatingsTest.loc[movie] += \
                                        np.dot(TopKSimilarityScore, \
                                               self.UserMovieMatrixTest.T\
                                                   .loc[TopKSimilarMovies])\
                                                     / TopKSimilarityScoreNorm
            PredictedRatingsTrain = PredictedRatingsTrain.T
            PredictedRatingsTest = PredictedRatingsTest.T
        return PredictedRatingsTrain, PredictedRatingsTest             

    def ComputeMSE(self,Actuals,Predictions):
        if isinstance(Actuals, pd.DataFrame):
            Actuals = Actuals.to_numpy()
        if isinstance(Predictions, pd.DataFrame):
            Predictions = Predictions.to_numpy()
        return mean_squared_error(Actuals[np.nonzero(Actuals)], \
                                  Predictions[np.nonzero(Actuals)])

    def TrainModel(self):
        Metrics = []
        self.ResultTrainMSE = []
        self.ResultTestMSE = []
        
        if self.modelMode == 'TRAIN':
            print('Training the Collaborative Filtering Model for different values of K neighbours')
            # print('...  Trying different values of K (Neighbours) to Train the model')
            # K_Trial = [8,32,64,128,256,512]
            k_Trial = [0,8,16,32,64,128,256]
            for k in k_Trial:
                self.k = k
                self.PredictedRatingsTrain, self.PredictedRatingsTest = self.PredictRating()
                TrainMSE = self.ComputeMSE(self.UserMovieMatrixTrain, \
                                           self.PredictedRatingsTrain)
                TestMSE = self.ComputeMSE(self.UserMovieMatrixTest, \
                                          self.PredictedRatingsTest)
                self.ResultTrainMSE.append(TrainMSE)
                self.ResultTestMSE.append(TestMSE)
                print("...Training for k = %d | Train MSE = %.4f | Test MSE = %.4f" \
                                  % (self.k, TrainMSE, TestMSE))
                TrainingMetrics = {'ModelName':self.GetModel(),
                                   'ModelMode':self.modelMode,
                                   'kNeighbours':self.k,
                                   'TrainMSE':TrainMSE,
                                   'TestMSE':TestMSE}
                Metrics.append(TrainingMetrics)
            
            # --------------------------------------------------------------------
            # Evaluate the Optimum Value of k Neighbours in Training
            self.Metrics = pd.DataFrame(Metrics)
            OptimumModelParameters = self.Metrics.loc[self.Metrics\
                                                          .TrainMSE.round(3) \
                                                      == min(self.Metrics.TrainMSE\
                                                                 .round(3))]\
                                         .sort_values(by = 'TestMSE')\
                                                 [['kNeighbours','TestMSE',\
                                                   'TrainMSE']]\
                                         .reset_index().iloc[0]
            print('\nOptimum Model Parameters are as follows:')
            print('....  Number of Optimum Neighbours (k) = '\
                  ,OptimumModelParameters.kNeighbours)
            #print('....  Number of Training Iterations (i) = '\
            #      ,OptimumModelParameters.Iteration)
            print('....  Results -> Optimum Train MSE:%.4f | Optimum Test MSE:%.4f'\
                  % (OptimumModelParameters.TrainMSE,\
                     OptimumModelParameters.TestMSE))    
                        
            # --------------------------------------------------------------------
            # Reset the Parameters and then Proceed to FIT the model
            self.k = int(OptimumModelParameters.kNeighbours)
            self.modelMode = 'FIT'
            print('....  Fitting the model based on %d Nearest Neighbours' % (self.k))
            self.TrainModel()
            
        else:
            print('Fitting the Collaborative Filtering Model for k = %d' %(self.k))
            self.PredictedRatingsTrain, self.PredictedRatingsTest = self.PredictRating()
            TrainMSE = self.ComputeMSE(self.UserMovieMatrixTrain, \
                                       self.PredictedRatingsTrain)
            TestMSE = self.ComputeMSE(self.UserMovieMatrixTest, \
                                      self.PredictedRatingsTest)
            self.ResultTrainMSE.append(TrainMSE)
            self.ResultTestMSE.append(TestMSE)
            
            if self.k == 0:
                k = None
            else:
                k = self.k
            print("...Fitted Model for k = %d | Train MSE = %.4f | Test MSE = %.4f" \
                                  % (self.k, TrainMSE, TestMSE))
            TrainingMetrics = {'ModelName':self.GetModel(),
                               'ModelMode':self.modelMode,
                               'kNeighbours':k,
                               'TrainMSE':TrainMSE,
                               'TestMSE':TestMSE}
            Metrics.append(TrainingMetrics)
            self.Metrics = pd.DataFrame(Metrics)
        
        return self
    
    def GetModel(self):
        if (self.CFMethod.upper() == 'USER'):
            # Condition for USER-USER CF
            MODEL = 'Memory Based User-User-CF | k = ' + str(self.k)
        else:
            # Condition for ITEM-ITEM CF
            MODEL = 'Memory Based Item-Item-CF | k = ' + str(self.k)
        return MODEL
                                      
    # Get Already rated movies that need to be excluded from recommendations
    def GetUserRatedMovies(self, UserId, ratings):
        RatedMovies = ratings.set_index('userId').loc[UserId]['movieId']
        return RatedMovies

    def RecommendMovies(self,UserId,RecommendForTest = True):
        #Predict User Ratings
        if RecommendForTest:
            PredictedRatings = self.PredictedRatingsTest
            ratings = self.movieData.ratingsTest
        else:
            PredictedRatings = self.PredictedRatingsTrain
            ratings = self.movieData.ratingsTrain
        
        PredictedUserRatings = PredictedRatings.loc[UserId]\
                                               .reset_index()\
                                               .rename(columns={'index':'movieId',\
                                                     UserId:'PredictedRating'})
        # Exclusion List of all movies that user has already rated in TEST
        ExcludeList = self.GetUserRatedMovies(UserId, ratings).tolist()
        
        UserRecommendations = PredictedUserRatings[~PredictedUserRatings['movieId']\
                                                   .isin(ExcludeList)]
        
        UserRecommendations = UserRecommendations\
                                .merge(self.movieData\
                                           .movies[['movieId','title','genres']],\
                                       how = 'left',\
                                       left_on = 'movieId',\
                                       right_on = 'movieId')\
                                .drop_duplicates()\
                                .sort_values(by = ['PredictedRating']\
                                                 ,ascending = False)\
                                .head(self.TOP_N)
        
        return UserRecommendations[['movieId','title','PredictedRating']] 

    

    