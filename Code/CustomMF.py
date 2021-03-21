# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:07:55 2021

@author: mashimpi
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class CustomMF:
    
    """ 
    ---------------- Matrix Factorization Technique ---------------------------
    
    Aim is to reduce the User-Item Rating Matrix into a lower dimension Matrix
    For 'n' Users and 'm' movies, assume that there are 'k' latent features  
    then,
        the RATING Matrix 'R' can be prected as
            R _(n x m) = X_(n x k) x Y_(k x m), where:
            X is a (nxk) matrix of 'n' users and 'k' latent features
            Y is a (kxm) matrix of 'k' latent features and 'm' movies
            
        In order to arrive at the best value of 'k' (number of Latent Factors)
        we will MINIMIZE the Cost Function
            C = R^2 - (X.Y)^2 + LambdaX * Summation(NormX) + LambdaY * Summation(NormY)
        The above Cost equations can be minimized using 2 methods:
            a. Alternate Least Squares (ALS) 
            b. Stocastic Gradient Descent
            
    --------------------------------------------------------------------------
            
    """     
    
    TOP_N = 5000
    
    def __init__(self,
                 movieData,
                 NumIterations = 50,
                 K = 40, # of latent Factors
                 OptimizationMethod = 'SGD',
                 LambdaUser = 0.001, # User Regularization
                 LambdaMovie = 0.001, # Movie Regularization
                 AlphaLearningRate = 0.001, # Learning Rate for SGD
                 BetaRegParameter = 0.01, # Regularization Parameter for SGD
                 modelMode = 'Fit'
                 ):
        
        print('Constructing Custom Matrix Factorization Model ...')
        self.movieData = movieData
        self.NumIterations = NumIterations
        self.K = K
        self.OptimizationMethod = OptimizationMethod.upper()
        self.modelMode = modelMode.upper()
        self.UserMovieMatrixTrain = self.movieData.UserMovieMatrixTrain.fillna(0)
        self.UserMovieMatrixTest = self.movieData.UserMovieMatrixTest.fillna(0)
        self.NumUsers = self.UserMovieMatrixTrain.index.nunique()
        self.NumMovies = self.UserMovieMatrixTrain.columns.nunique()
        
        # ALS Parameters
        self.LambdaUser = LambdaUser
        self.LambdaMovie = LambdaMovie
        
        # SGD Parameters
        self.AlphaLearningRate = AlphaLearningRate
        self.BetaRegParameter = BetaRegParameter
        
        # Train and Fit the Model
        if self.OptimizationMethod == 'SGD':
            self.Reg = self.BetaRegParameter
        else:
            self.Reg = self.LambdaUser
        print('Parameters -> Optimization Method = %s | Regularization = %.4f | Learning Rate = %.4f\n' \
              % (self.OptimizationMethod,self.Reg, self.AlphaLearningRate))
        self.TrainModel()
        #self.modelMode = 'FIT'
        #self.TrainModel = self.TrainModel()
        self.PredictedRatingsTrain = self.PredictedRatings
        self.PredictedRatingsTest = self.PredictedRatings
        
    
    def GetModel(self):
        if self.OptimizationMethod == 'ALS':
            MODEL = 'Matrix Factorization (ALS) - ' + str(self.K) + ' Latent Features'
        else:
            MODEL = 'Matrix Factorization (SGD) - ' + str(self.K) + ' Latent Features'
        return MODEL
    
    def ComputeALS(self,
                   UserMovieMatrix,
                   VectorToSolve,
                   FixedVector,
                   Lambda
                   ):
        
        """ 
            Differentiating both sides of COST FUNCTION w.r.t X & Y, 
            the final User-Feature-Matrix (X) or Item-Feature-Matrix (Y) is 
            computed as:
                X = R * Y * InverseOf ( Y * Y.T + Lambda_X * I)
                Y = R * X * InverseOf ( X.T * X + Lambda_Y * I)
            If MF Method is USER, 
                compute FeatureVector(X) = R * Y * InverseOf ( YT*Y + Lambda_X * I)
            If MF Method is ITEM,
                compute FeatureVector(Y) = R * X * InverseOf ( X*X + Lambda_Y * I)
            
            FeatureVector = R * FixedVector * InverseOf ( FixedVector.FixedVector 
                                                         + Lambda * I)
                Lets break RHS into 2 parts
                    RHS = Part 1 * Part 2
                    where,
                        Part 1 = R * FixedVector 
                        Part 2 = Inverse Of (Solved Vector + Lambda_X * I)
                    
            """
        RHSPart1 = UserMovieMatrix.dot(FixedVector)
        RHSPart2 = np.linalg.inv(FixedVector.T.dot(FixedVector) \
                                     + np.eye(self.K) * Lambda)
        return RHSPart1.dot(RHSPart2)
    
    def ComputeSGD(self):
        
        for user, movie, rating in self.TrainingDataSet:
            
            # Compute predicted Rating for the given user,movie
            predictedRating = self.BiasGlobal + self.BiasUser[user] \
                + self.BiasMovie[movie] + self.UserFeatureMatrix[user,:]\
                                              .dot(self.MovieFeatureMatrix[movie,:].T)
            
            # Compute the Error between Actual and Predicted Rating
            error = (rating - predictedRating)

            # Update User and Movie Biases
            self.BiasUser[user] += self.AlphaLearningRate \
                * (error - self.BetaRegParameter * self.BiasUser[user])
            
            self.BiasMovie[movie] += self.AlphaLearningRate \
                * (error - self.BetaRegParameter * self.BiasMovie[movie])
            
            # Update User and Movie Latent Feature Matrices
            self.UserFeatureMatrix[user,:] += self.AlphaLearningRate \
                * (error * self.MovieFeatureMatrix[movie,:] \
                   - self.BetaRegParameter * self.UserFeatureMatrix[user,:])
            
            self.MovieFeatureMatrix[movie,:] += self.AlphaLearningRate \
                * (error * self.UserFeatureMatrix[user,:] \
                   - self.BetaRegParameter * self.MovieFeatureMatrix[movie,:])
                    
            return self
            
    def PredictRatings(self):
        # Predict Ratings for every User and Movie 
        return self.UserFeatureMatrix.dot(self.MovieFeatureMatrix.T)
    
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
            print('Training the Matrix Factorization Model over %d iterations' % (\
              self.NumIterations))
            print('Trying different values of K (Latent Features) to Train the model')
            #K_Trial = [8,32,64,128,256,512]
            K_Trial = [8,32,64,128,256]
        else:
            K_Trial = [int(self.K)]
            print('Fitting the Matrix Factorization Model for %d Latent Features over %d iterations' % (self.K,self.NumIterations))
        
        for KTrial in K_Trial:
            self.K = KTrial
            #self.UserFeatureMatrix = np.random.random((self.NumUsers, self.K))
            #self.MovieFeatureMatrix = np.random.random((self.NumMovies, self.K))
            self.UserFeatureMatrix = np.random.normal(scale=1./self.K,\
                                                      size=(self.NumUsers, self.K))
            self.MovieFeatureMatrix = np.random.normal(scale=1./self.K,\
                                                      size=(self.NumMovies, self.K))
            
            if self.OptimizationMethod == 'ALS': 
                print('Trying ALS with %d Latent Features ...' %(self.K))
                for i in range(self.NumIterations):
                    self.UserFeatureMatrix = self.ComputeALS(self.UserMovieMatrixTrain, \
                                                             self.UserFeatureMatrix, \
                                                             self.MovieFeatureMatrix, \
                                                             self.LambdaUser)
                    self.MovieFeatureMatrix = self.ComputeALS(self.UserMovieMatrixTrain.T, \
                                                             self.MovieFeatureMatrix, \
                                                             self.UserFeatureMatrix, \
                                                             self.LambdaMovie) 
                    self.PredictedRatings = self.PredictRatings()
                    
                    TrainMSE = self.ComputeMSE(self.UserMovieMatrixTrain, \
                                               self.PredictedRatings)
                    TestMSE = self.ComputeMSE(self.UserMovieMatrixTest, \
                                              self.PredictedRatings)
                    self.ResultTrainMSE.append(TrainMSE)
                    self.ResultTestMSE.append(TestMSE)
                    if (i+1) % 10 == 0:
                        print("... Iteration: %d | Train MSE = %.4f | Test MSE = %.4f" \
                              % (i+1, TrainMSE, TestMSE))
                    TrainingMetrics = {'ModelName':self.GetModel(),
                                       'ModelMode':self.modelMode,
                                       'OptimizationMethod':self.OptimizationMethod,
                                       'NumLatentFeatures':self.K,
                                       'Iteration':i,
                                       'TrainMSE':TrainMSE,
                                       'TestMSE':TestMSE}
                    Metrics.append(TrainingMetrics)
            else:
                # Code for SGD
                print('Computing SGD with %d Latent Features ...' %(self.K))
                self.BiasUser = np.zeros(self.NumUsers)
                self.BiasMovie = np.zeros(self.NumMovies)
                self.BiasGlobal = np.mean(self.UserMovieMatrixTrain.to_numpy()\
                                          [np.where(self.UserMovieMatrixTrain\
                                                        .to_numpy() != 0)\
                                          ])
                
                    
                self.TrainingDataSet = [(user, movie, self.UserMovieMatrixTrain\
                                                          .to_numpy()[user, movie])\
                                        for user in range(self.NumUsers)\
                                            for movie in range(self.NumMovies)\
                                                if self.UserMovieMatrixTrain\
                                                       .to_numpy()[user, movie] > 0]
                    
                for i in range(self.NumIterations):
                    
                    # Randomly take the Actual Ratings and then compute gradient
                    np.random.shuffle(self.TrainingDataSet)
                    self.ComputeSGD()
                    
                    """ 
                    Based on Gradient, the User-Feature and Movie-Feature Cells
                    are computed. Then we take dot product and predict the rating
                    for the particular user & movie.
                    
                    Next, we need to compute this for the complete Matrices of
                    User-LatentFeatures and Movie-LatentFeatures
                    
                    """
                    self.PredictedRatings = self.BiasGlobal \
                                                + self.BiasUser[:,np.newaxis] \
                                                + self.BiasMovie[np.newaxis:,] \
                                                + self.UserFeatureMatrix\
                                                      .dot(self.MovieFeatureMatrix.T)
                    
                    TrainMSE = self.ComputeMSE(self.UserMovieMatrixTrain, \
                                               self.PredictedRatings)
                    TestMSE = self.ComputeMSE(self.UserMovieMatrixTest, \
                                              self.PredictedRatings)
                    
                    self.ResultTrainMSE.append(TrainMSE)
                    self.ResultTestMSE.append(TestMSE)
                    
                    if (i+1) % 25 == 0:
                        print("... Iteration: %d | Train MSE = %.4f | Test MSE = %.4f" \
                              % (i+1, TrainMSE, TestMSE))
                        
                    TrainingMetrics = {'ModelName':self.GetModel(),
                                       'ModelMode':self.modelMode,
                                       'OptimizationMethod':self.OptimizationMethod,
                                       'NumLatentFeatures':self.K,
                                       'Iteration':i,
                                       'TrainMSE':TrainMSE,
                                       'TestMSE':TestMSE}
                    Metrics.append(TrainingMetrics)
                self.PredictedRatings = pd.DataFrame(self.PredictedRatings,\
                                                     index = self.UserMovieMatrixTrain.index,
                                                     columns = self.UserMovieMatrixTrain.columns)
        self.Metrics = pd.DataFrame(Metrics)
        
        
        
        # Get the most Optimum Parameters and then FIT the model
        if self.modelMode == 'TRAIN':
            OptimumModelParameters = self.Metrics\
                                  .loc[self.Metrics.TrainMSE.round(3) \
                                       == min(self.Metrics.TrainMSE.round(3))]\
                                  .sort_values(by = 'TestMSE')\
                                      [['NumLatentFeatures','Iteration',\
                                        'TestMSE','TrainMSE']]\
                                  .reset_index().iloc[0]
            print('\nOptimum Model Parameters are as follows:')
            print('....  Number of Latent Features (K) = '\
                  ,OptimumModelParameters.NumLatentFeatures)
            #print('....  Number of Training Iterations (i) = '\
            #      ,OptimumModelParameters.Iteration)
            print('....  Results -> Optimum Train MSE:%.4f | Optimum Test MSE:%.4f'\
                  % (OptimumModelParameters.TrainMSE,\
                     OptimumModelParameters.TestMSE))    
            
            # Reset the Parameters and then Proceed to FIT the model
            self.K = int(OptimumModelParameters.NumLatentFeatures)
            self.NumIterations = 100 
            self.modelMode = 'FIT'
            print('....  Fitting the model based on %d Latent Features for %d Iterations' \
                  % (self.K,self.NumIterations))
            self.TrainModel()
            
            
        
        # print('Least Value of MSE (Train): ',min(self.ResultTrainMSE))
        # print('Least Value of MSE (Test): ',min(self.ResultTestMSE))
        
        # Plot the learning curve
        # linewidth = 3
        # plt.title('MSE by Iteration for %s - MATRIX FACTORIZATION (Mode:%s)' \
        #           % (self.OptimizationMethod, self.modelMode))
        # plt.plot(self.ResultTestMSE, label = 'Test', linewidth = linewidth)
        # plt.plot(self.ResultTrainMSE, label = 'Train', linewidth = linewidth)
        # plt.xlabel('Learning Iterations')
        # plt.ylabel('MSE')
        # plt.legend(loc = 'best')
        
        return self
    
    # Get Already rated movies that need to be excluded from recommendations
    def GetUserRatedMovies(self, UserId, ratings):
        RatedMovies = ratings.set_index('userId').loc[UserId]['movieId']
        return RatedMovies

    def RecommendMovies(self,UserId,RecommendForTest = True):
        #Predict User Ratings
        if RecommendForTest:
            ratings = self.movieData.ratingsTest
        else:
            ratings = self.movieData.ratingsTrain
        
        PredictedUserRatings = self.PredictedRatings.loc[UserId]\
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
           
    
       
    
        
        
        
    
    
    
    
        
        
    
        
        
        
        
        
        
        
        