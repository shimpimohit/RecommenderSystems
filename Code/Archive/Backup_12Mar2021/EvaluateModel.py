# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:38:13 2021

@author: mashimpi
"""

# ------------- EVALUATION METRICS FOR RECOMMENDER SYSTEMS ----------------#
# This class defines various Metrics to Evaluate various Recommender Models

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from CFModel import CFModel
from MFModel import MFModel
from PopularityModel import PopularityModel
from ContentModel import ContentModel


class EvaluateModel:
    
    # Defining some Class Constants
    
    SAMPLE_SIZE = 100 # This is the Sample Sixe for Evaluating Recommedations
    RANDOM_SEED = 100 # Setting some constant Seed Value for Sampling
    THRESHOLD_CUTOFF = 4.0 # This is the Rating Cutoff THreshold for Cumilative Hit Rate
    
    def __init__(self, movies,ratings,modelName,evaluateForTest = True,\
                 method = 'User', k = 0, EliminateBias = True):
        self.movies = movies
        self.ratings = ratings 
        self.evaluateForTest = evaluateForTest
        # Note: The 'Test' parameter determines whether the model 
        # evaluates on TEST dataset (Default) or TRAIN dataset
        # If initialed as Test = True, then model is evaluated for TEST SET
        # If initialed as Test = False, then model is evaluated for TRAINING SET
        
        self.modelName = modelName
        # This is the Model - CF / Content / MF / etc.
        
        # These below are only applicable for Collaborative Filtering
        self.method = method
        self.k = k
        self.EliminateBias = EliminateBias
        
        # Define Training / Testing Set
        if self.evaluateForTest:
            # TestSet = ratingsTest
            # TrainSet = ratingsAntiTestSet
            self.ratingsAntiTest,self.ratingsTest = train_test_split(ratings,\
                                                    stratify = ratings['userId'],\
                                                    test_size = 0.2,\
                                                    random_state = self.RANDOM_SEED)
        else:
            # if EvaluateForTest = False, Swap the Test Set and TrainSet
            self.ratingsTest,self.ratingsAntiTest = train_test_split(ratings,\
                                                    stratify = ratings['userId'],\
                                                    test_size = 0.2,\
                                                    random_state = RANDOM_SEED)
        
        self.ratingsTestIndexed = self.ratingsTest.set_index('userId') 
        self.ratingsAntiTestIndexed = self.ratingsAntiTest.set_index('userId') 
        
        #self.model = self.InitializeModel(self)
        
    def InitializeModel(self,ratingsDataSet):
        if self.modelName == "CFModel":
            return CFModel(self.movies,ratingsDataSet,self.method,\
                           self.k,self.EliminateBias)
        elif self.modelName == "ContentModel":
            return ContentModel(self.movies,self.ratingsTest)
        elif self.modelName == "MFModel":
            return MFModel(movies,ratings)
        else:
            raise Exception("Model Not Recognised")
    
    def GetUserRatedMovies(self, UserId):
        return self.ratingsTest.set_index('userId').loc[UserId]['movieId']
    
    def GenerateLOOCVTrainDataSet(self,LeaveOneOutIndex):
        # This generates a dataset of all Ratings except the
        # one that needs to be left out intentionally for Hit Rate
        return self.ratingsTest.loc[[i for i in self.ratingsTest.index \
                                               if i != LeaveOneOutIndex]]
    
    def GetUserNotRatedSample(self, UserId):
        random.seed(self.RANDOM_SEED)
        RatedMovies = set(self.ratings.set_index('userId').loc[UserId]['movieId'])
        
        AllMovies = set(self.ratings['movieId'])
        
        NotRatedMovies = (AllMovies - RatedMovies)
        NotRatedSample = set(random.sample(NotRatedMovies,self.SAMPLE_SIZE))
        return NotRatedSample
    
    def VerifyHit(self,MovieId,Recommendations,N):
        try:
            #Index = next(counter for counter, \
            #             movie in enumerate(Recommendations) \
            #                 if movie == MovieId)
            Index = Recommendations.to_list().index(MovieId)
        except:
            Index = -1
        hit = int(Index in range(0,N))
        return hit, Index
    
    def IsHit(self,UserId,N):
        TopNRecommendedMovies = self.model.RecommendMovies(UserId)['movieId'][:N]
        RatedMovies = self.GetUserRatedMovies(UserId)
        if sum(TopNRecommendedMovies.isin(RatedMovies)) > 0:
            hit = 1
        else:
            hit = 0
        return hit
    
    def EvaluateRecommenderforUser(self,UserId):
        # Get Movies Rated by the User from the Testing DataSet
        UserTestDataSet = self.ratingsTestIndexed.loc[UserId]
        UserTestSetCount = UserTestDataSet.shape[0]
        UserTestSetAboveThresholdCount = UserTestDataSet[UserTestDataSet['rating'] \
                                                       >= self.THRESHOLD_CUTOFF]\
                                                    .shape[0]
        
        # Evaluate the following Top-N Mettics using Leave-One-Out CrossValidation
        #    a. Hit Rate (a.k.a Recall) @ 5
        #    b. Hit Rate (a.k.a Recall) @ 10
        #    c. Cumilative Hit Rate
        #    d. Average Reciprocal Hit Rate
        
        Hit5Count, Hit10Count = 0,0
        cHitCount = 0
        SumOfReciprocalRank = 0
        for row in UserTestDataSet.itertuples():
            # Generate Leave-One-Out SubSet from the Test Set
            LOOTestSet = self.GenerateLOOCVTrainDataSet(row[0])
            MovieId = row[2] # Third column of the Ratings Dataset
            MovieRating = row[3] # Fourth column is the Rating
            
            #Generate Recommendations with this New Test Set without a movie
            Model = self.InitializeModel(LOOTestSet)
            RecommendedMovies = Model.RecommendMovies(UserId)['movieId']
            
            # Compute Hits@5 and Hits@10
            Hit5 ,Rank = self.VerifyHit(MovieId, RecommendedMovies, 5)
            Hit10,Rank = self.VerifyHit(MovieId, RecommendedMovies, 10)
            Hit5Count  += Hit5
            Hit10Count += Hit10
            
            # Compute Hit Count above Threshold for Cumilative Hit Rate
            if MovieRating >= self.THRESHOLD_CUTOFF and Rank > -1:
                cHitCount += 1
            
            # Compute sum of Reciprocal Ranks
            if Rank > -1:
                SumOfReciprocalRank += 1 / Rank
        
        # Evaluate Hit Rate (or Recall) Metric
        Recall5  = Hit5Count /float(UserTestSetCount)
        Recall10 = Hit10Count/float(UserTestSetCount) 
        
        # Evaluate Cumilative Hit Rate Metric
        CumilativeHitRate = cHitCount/float(UserTestSetAboveThresholdCount)
        
        # Evaluate Average Reciprcal Hit Rate
        AvgReciprocalHitRate = SumOfReciprocalRank / float(UserTestSetCount)
        
        UserMetricsList = [Model.GetModel(),\
                           Hit5Count,\
                           Hit10Count,\
                           UserTestSetCount,\
                           cHitCount,\
                           UserTestSetAboveThresholdCount,\
                           Recall5,\
                           Recall10,\
                           CumilativeHitRate,\
                           SumOfReciprocalRank,\
                           AvgReciprocalHitRate]
        
        UserMetrics = {'Model Name':Model.GetModel(),
                       'Hit5 Count':Hit5Count, 
                       'Hit10 Count':Hit10Count, 
                       'Count of Ratings': UserTestSetCount,
                       'Cumilative Hit Count':cHitCount,
                       'Count of Ratings above Threshold':UserTestSetAboveThresholdCount,
                       'Recall5': Recall5,
                       'Recall10': Recall10,
                       'Cumilative Hit Rate':CumilativeHitRate,
                       'Sum of Reciprocal Rank':SumOfReciprocalRank,
                       'AvgReciprocalHitRate':AvgReciprocalHitRate
                       }
        return UserMetrics     
    
    # ---- Code Below to be commented ----#
    # def EvaluateRecommenderforUser(self,UserId):
    #     # Get Movies Rated by the User from the Testing DataSet
    #     #UserTestSet = set(self.ratingsTestIndexed.loc[UserId]['movieId'])
    #     UserTestSet = set(self.ratingsTestIndexed.loc[UserId]['movieId'])
    #     UserTestSetCount = len(UserTestSet)
        
        
        
    #     # Generate Recommendations
    #     UserRecommendations = self.model.RecommendMovies(UserId)
        
        
    #     # Evaluate Metrics 
    #     hit5Count, hit10Count = 0,0
    #     for movie in UserTestSet:
    #         # METRIC 1 - HIT RATE
    #         #---------------------
    #         # Get Random Sample of Not Rated Movies
    #         NotRatedSample = self.GetUserNotRatedSample(UserId)
            
    #         # Combine Sample with the iterated Movie
    #         setMovie = {movie}
    #         CombinedSet = NotRatedSample.union(setMovie)
            
    #         # Separate out Recommendations that are either in the Rated Sample
    #         # OR in the Random Sample of Not Rated Movies
    #         ValidatedRecommendations = \
    #             UserRecommendations[UserRecommendations['movieId']\
    #                                 .isin(CombinedSet)]['movieId'].values
            
    #         hit5, index5 = self.VerifyHit(movie, ValidatedRecommendations, 5)
    #         #print('hit5:',hit5,'index5:',index5)        
    #         hit5Count += hit5
    #         hit10, index10 = self.VerifyHit(movie, ValidatedRecommendations, 10)
    #         hit10Count += hit10
            
    #         # METRIC #2 - MAE
    #         #------------------
    #         #if self.model.GetModel() != 'Content-Based':
    #             #UserRecommendations
                
        
    #     # Evaluate Recall Metric
    #     recall5 = hit5Count/float(UserTestSetCount)
    #     recall10 = hit10Count/float(UserTestSetCount)
        
    #     # Combine all Metrics for a User
    #     UserMetrics = {'Model Name':self.model.GetModel(),
    #                    'hit5 Count':hit5Count, 
    #                    'hit10 Count':hit10Count, 
    #                    'Count of Ratings': UserTestSetCount,
    #                    'recall5': recall5,
    #                    'recall10': recall10}
        
    #     return UserMetrics
        
    def EvaluateRecommenderModel(self):
        # Create a list to store all User Metrics
        UserMetrics = []
        
        # Generate Predictions
        try:
            ActualRatings = self.ratingsTest.pivot_table(index='userId',\
                                                       columns='movieId',\
                                                       values='rating')\
                                            .to_numpy()
            ActualRatings = ActualRatings[~np.isnan(ActualRatings)].flatten()
            
            PredictedRatings = self.model.PredictRating().to_numpy()
            PredictedRatings = PredictedRatings[~np.isnan(ActualRatings)].flatten()
            
            ModelMSE = mean_squared_error(PredictedRatings, ActualRatings)
            ModelMAE = mean_absolute_error(PredictedRatings, ActualRatings)
        except:
            ModelMAE = None
            ModelMSE = None
        
        
        for Index,UserId in enumerate(list(self.ratingsTestIndexed\
                                           #self.ratings.set_index('userId')\
                                               .index.unique().values)):
            UserMetricforModel = self.EvaluateRecommenderforUser(UserId)
            UserMetricforModel['userId'] = UserId
            UserMetrics.append(UserMetricforModel)
        
        EvaluationResultsforUser = pd.DataFrame(UserMetrics)\
                                     .sort_values('Count of Ratings',\
                                                  ascending = False)
        
        print('debug - Evaluation Results')
        print(EvaluationResultsforUser.head(10))
        ModelRecall5 = EvaluationResultsforUser['hit5 Count'].sum() \
                         / float(EvaluationResultsforUser['Count of Ratings'].sum())
        ModelRecall10 = EvaluationResultsforUser['hit10 Count'].sum() \
                         / float(EvaluationResultsforUser['Count of Ratings'].sum())
                         
        ModelMetrics = {'Model Name': self.model.GetModel(),
                        'Recall5':ModelRecall5,
                        'Recall10':ModelRecall10,
                        'MAE':ModelMAE,
                        'MSE':ModelMSE}
        
        return ModelMetrics,EvaluationResultsforUser
                                     
        
        
            
        
        
    
    
        
        