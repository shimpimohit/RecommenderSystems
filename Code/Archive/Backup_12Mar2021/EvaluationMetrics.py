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
from EvaluationMetrics import EvaluationMetrics

class EvaluateModel:
    
    # Defining some constants
    
    SAMPLE_SIZE = 100 # This is the Sample Sixe for Evaluating Recommedations
    RANDOM_SEED = 100 # Setting some constant Seed Value for Sampling
    
    def __init__(self, movies,ratings,modelName,evaluateForTest = True\
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
        if EvaluateForTest:
            # TestSet = ratingsTest
            # TrainSet = ratingsAntiTestSet
            self.ratingsAntiTest,self.ratingsTest = train_test_split(ratings,\
                                                    stratify = ratings['userId'],\
                                                    test_size = 0.2,\
                                                    random_state = RANDOM_SEED)
        else:
            # if EvaluateForTest = False, Swap the Test Set and TrainSet
            self.ratingsTest,self.ratingsAntiTest = train_test_split(ratings,\
                                                    stratify = ratings['userId'],\
                                                    test_size = 0.2,\
                                                    random_state = RANDOM_SEED)
        
        self.ratingsTestIndexed = self.ratingsTest.set_index('userId') 
        self.ratingsAntiTestIndexed = self.ratingsAntiTest.set_index('userId') 
        
        self.model = self.InitializeModel(self)
        
    def InitializeModel(self, movies, ratings, Method = self.method, k=self.k, EliminateBias):
        if name == "CFModel":
            return CFModel(self.movies,self.ratingsTest,self.method,\
                           self.k,self.EliminateBias)
        elif name == "ContentModel":
            return ContentModel(self.movies,self.ratingsTest)
        elif name == "MFModel":
            return MFModel(movies,ratings)
        else:
            raise Exception("Model Not Recognised")
    
    def GetUserRatedMovies(self, UserId):
        return self.ratingsTest.set_index('userId').loc[UserId]['movieId']
    
    def GetUserNotRatedSample(self, UserId):
        random.seed(self.RANDOM_SEED)
        RatedMovies = set(self.ratings.set_index('userId').loc[UserId]['movieId'])
        
        AllMovies = set(self.ratings['movieId'])
        
        NotRatedMovies = (AllMovies - RatedMovies)
        NotRatedSample = set(random.sample(NotRatedMovies,self.SAMPLE_SIZE))
        return NotRatedSample
    
    def VerifyHit(self,MovieId,Recommendations,N):
        try:
            Index = next(counter for counter, \
                         movie in enumerate(Recommendations) \
                             if movie == MovieId)
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
        #UserTestSet = set(self.ratingsTestIndexed.loc[UserId]['movieId'])
        UserTestSet = set(self.ratingsTestIndexed.loc[UserId]['movieId'])
        UserTestSetCount = len(UserTestSet)
        
        
        
        # Generate Recommendations
        UserRecommendations = self.model.RecommendMovies(UserId)
        
        
        # Evaluate Metrics 
        hit5Count, hit10Count = 0,0
        for movie in UserTestSet:
            # METRIC 1 - HIT RATE
            #---------------------
            # Get Random Sample of Not Rated Movies
            NotRatedSample = self.GetUserNotRatedSample(UserId)
            
            # Combine Sample with the iterated Movie
            setMovie = {movie}
            CombinedSet = NotRatedSample.union(setMovie)
            
            # Separate out Recommendations that are either in the Rated Sample
            # OR in the Random Sample of Not Rated Movies
            ValidatedRecommendations = \
                UserRecommendations[UserRecommendations['movieId']\
                                    .isin(CombinedSet)]['movieId'].values
            
            hit5, index5 = self.VerifyHit(movie, ValidatedRecommendations, 5)
            #print('hit5:',hit5,'index5:',index5)        
            hit5Count += hit5
            hit10, index10 = self.VerifyHit(movie, ValidatedRecommendations, 10)
            hit10Count += hit10
            
            # METRIC #2 - MAE
            #------------------
            #if self.model.GetModel() != 'Content-Based':
                #UserRecommendations
                
        
        # Evaluate Recall Metric
        recall5 = hit5Count/float(UserTestSetCount)
        recall10 = hit10Count/float(UserTestSetCount)
        
        # Combine all Metrics for a User
        UserMetrics = {'ModelName':self.model.GetModel(),
                       'hit5Count':hit5Count, 
                       'hit10Count':hit10Count, 
                       'CountOfRatings': UserTestSetCount,
                       'recall5': recall5,
                       'recall10': recall10}
        
        return UserMetrics
        
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
                                     
        
        
            
        
        
    
    
        
        