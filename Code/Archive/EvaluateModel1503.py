# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:43:50 2021

@author: mashimpi
"""
import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class EvaluateModel:
    
    SAMPLE_SIZE = 100 # This is the Sample Sixe for Evaluating Recommedations
    RANDOM_SEED = 100 # Setting some constant Seed Value for Sampling
    THRESHOLD_CUTOFF = 4.0 # This is the Rating Cutoff THreshold for Cumilative Hit Rate
    
    def __init__(self,model, evaluateForTrain = True):
        
        self.model = model
        self.evaluateForTrain = evaluateForTrain 
        # by default model will always be evaluated on how it is trained
        
        if self.evaluateForTrain:
            self.ratingsTrain = self.model.movieData.ratingsTrain
            self.ratingsTest = self.model.movieData.ratingsTest
            self.DataSetName = 'Train'
            self.ActualRatings = self.model.UserMovieMatrixTrain
            self.PredictedRatings = self.model.PredictedRatingsTrain
        else:
            #Training DataSet becomes TEST Set and vice versa
            self.ratingsTrain = self.model.movieData.ratingsTest
            self.ratingsTest = self.model.movieData.ratingsTrain
            self.DataSetName = 'Test'
            self.ActualRatings = self.model.UserMovieMatrixTest
            self.PredictedRatings = self.model.PredictedRatingsTest
        
    def GetUserNotRatedSample(self, UserId):
        random.seed(self.RANDOM_SEED)
        RatedMovies = set(self.model.movieData.ratings.set_index('userId')\
                                                      .loc[UserId]['movieId'])
        
        AllMovies = set(self.model.movieData.ratings['movieId'])
        
        NotRatedMovies = (AllMovies - RatedMovies)
        NotRatedSample = set(random.sample(NotRatedMovies,self.SAMPLE_SIZE))
        return NotRatedSample
    
    def VerifyHit(self,MovieId,Recommendations,N):
        try:
            Index = Recommendations.to_list().index(MovieId)
        except:
            Index = -1
        hit = int(Index in range(0,N))
        return hit, Index
    
    def EvaluateRecommenderforUser(self,UserId):
        # Get Movies Rated by the User from the Testing DataSet
        UserEvalDataSet = self.ratingsTest.set_index('userId').loc[UserId]
        UserEvalDataSetCount = UserEvalDataSet.shape[0]
        
        UserEvalSetAboveThresholdCount = UserEvalDataSet[UserEvalDataSet['rating'] \
                                                       >= self.THRESHOLD_CUTOFF]\
                                                    .shape[0]
        
        # Generate Recommendations
        if self.evaluateForTrain: 
            UserRecommendations = self.model\
                                       .RecommendMovies(UserId,
                                                        RecommendForTest = False)
        else:
            UserRecommendations = self.model.RecommendMovies(UserId)
        
                                       
        # Evaluate the following Top-N Metrics using SAMPLING 
        #    a. Hit Rate (a.k.a Recall) @ 5
        #    b. Hit Rate (a.k.a Recall) @ 10
        #    c. Cumilative Hit Rate
        #    d. Average Reciprocal Hit Rate
        # Note: Evaluation Sample has 1 RATED Movie and 100 Non-Rated Movies
        #       Random Sample of 100 Non-Rated movies eliminates the Bias 
        #       of Users with too many/few ratings.
        
        Hit5Count, Hit10Count = 0,0
        cHitCount = 0
        SumOfReciprocalRank = 0
        
        for row in UserEvalDataSet.itertuples():
            MovieId = row[1] # Second column of the Ratings Dataset
            print 
            MovieRating = row[2] # Third column is the Rating
            
            # Get Random Sample of Not Rated Movies
            NotRatedSample = self.GetUserNotRatedSample(UserId)
            
            # Combine Sample with the iterated Movie
            setMovie = {MovieId}
            CombinedSet = NotRatedSample.union(setMovie)
            
            # Separate out Recommendations that are either in the Rated Sample
            # OR in the Random Sample of Not Rated Movies
            ValidatedRecommendations = \
                UserRecommendations[UserRecommendations['movieId']\
                                    .isin(CombinedSet)]['movieId'].values
                    
            # Compute Hits@5 and Hits@10
            Hit5 ,Rank = self.VerifyHit(MovieId, ValidatedRecommendations, 5)
            Hit10,Rank = self.VerifyHit(MovieId, ValidatedRecommendations, 10)
            Hit5Count  += Hit5
            Hit10Count += Hit10
            
            # Compute Hit Count above Threshold for Cumilative Hit Rate
            if MovieRating >= self.THRESHOLD_CUTOFF and Rank > -1:
                cHitCount += 1
            
            # Compute sum of Reciprocal Ranks
            if Rank > -1:
                SumOfReciprocalRank += 1 / Rank
                
        # Evaluate Hit Rate (or Recall) Metric
        Recall5  = Hit5Count /float(UserEvalDataSetCount)
        Recall10 = Hit10Count/float(UserEvalDataSetCount) 
        
        # Evaluate Cumilative Hit Rate Metric
        if UserEvalSetAboveThresholdCount != 0:
            CumilativeHitRate = cHitCount/float(UserEvalSetAboveThresholdCount)
        else:
            CumilativeHitRate = 0
        
        # Evaluate Average Reciprcal Hit Rate
        AvgReciprocalHitRate = SumOfReciprocalRank / float(UserEvalDataSetCount)
        
        UserMetricsList = [self.model.GetModel(),\
                           self.DataSetName,\
                           Hit5Count,\
                           Hit10Count,\
                           UserEvalDataSetCount,\
                           cHitCount,\
                           UserEvalSetAboveThresholdCount,\
                           Recall5,\
                           Recall10,\
                           CumilativeHitRate,\
                           SumOfReciprocalRank,\
                           AvgReciprocalHitRate]
        
        UserMetrics = {'ModelName':self.model.GetModel(),
                       'DataSet':self.DataSetName,
                       'Hit5Count':Hit5Count, 
                       'Hit10Count':Hit10Count, 
                       'CountOfRatings': UserEvalDataSetCount,
                       'CumilativeHitCount':cHitCount,
                       'RatingsAboveThresholdCount':UserEvalSetAboveThresholdCount,
                       'Recall5': Recall5,
                       'Recall10': Recall10,
                       'CumilativeHitRate':CumilativeHitRate,
                       'SumReciprocalRank':SumOfReciprocalRank,
                       'AvgReciprocalHitRate':AvgReciprocalHitRate
                       }
        return UserMetrics    
    
    def EvaluateRecommenderModel(self):
        # Create a list to store all User Metrics
        UserMetrics = []

        try:
            ActualRatings = self.ActualRatings.to_numpy()[self.ActualRatings\
                                                              .to_numpy()\
                                                              .nonzero()]
            
            PredictedRatings = self.PredictedRatings.to_numpy()[self.ActualRatings\
                                                              .to_numpy()\
                                                              .nonzero()]
            ModelMSE = mean_squared_error(PredictedRatings, ActualRatings)
            ModelMAE = mean_absolute_error(PredictedRatings, ActualRatings)
        except:
            ModelMAE = None
            ModelMSE = None

        for Index,UserId in enumerate(list(self.ratingsTest.set_index('userId')\
                                               .index.unique()\
                                               .sort_values()\
                                               .to_numpy())):
            print('UserId :',UserId)
            UserMetricforModel = self.EvaluateRecommenderforUser(UserId)
            UserMetricforModel['userId'] = UserId
            UserMetrics.append(UserMetricforModel)
        
        EvaluationResultsforUser = pd.DataFrame(UserMetrics)\
                                     .sort_values('CountOfRatings',\
                                                  ascending = False)
        
        print('debug - Evaluation Results')
        print(EvaluationResultsforUser.head(10))
        ModelRecall5 = EvaluationResultsforUser['Hit5Count'].sum() \
                         / float(EvaluationResultsforUser['CountOfRatings'].sum())
        ModelRecall10 = EvaluationResultsforUser['Hit10Count'].sum() \
                         / float(EvaluationResultsforUser['CountOfRatings'].sum())
                         
        ModelMetrics = {'ModelName': self.model.GetModel(),
                        'DataSet':self.DataSetName,
                        'Recall5':ModelRecall5,
                        'Recall10':ModelRecall10,
                        'MAE':ModelMAE,
                        'MSE':ModelMSE}
        
        return ModelMetrics,EvaluationResultsforUser
        