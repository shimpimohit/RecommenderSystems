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

class EvaluationMetrics:
    
    # Defining some constants
    
    SAMPLE_SIZE = 100 # This is the Sample Sixe for Evaluating Recommedations
    RANDOM_SEED = 100 # Setting some constant Seed Value for Sampling
    
    def __init__(self, movies,ratings,ratingsTest,model):
        self.movies = movies
        #self.ratingsTrain = ratingsTrain
        #self.ratingsTest = ratingsTest
        self.ratings = ratings
        self.ratingsTest = ratingsTest
        #self.ratingsTrain,self.ratingsTest = train_test_split(ratings,\
        #                                         stratify = ratings['userId'],\
        #                                         test_size = 0.2,\
        #                                         random_state = self.RANDOM_SEED) 
        #self.ratings = pd.concat([ratingsTrain, ratingsTest], ignore_index=True)
        #self.ratingsTrainIndexed = self.ratingsTrain.set_index('userId')    
        self.ratingsTestIndexed = self.ratingsTest.set_index('userId') 
        self.model = model
        
    def GetUserRatedMovies(self, UserId):
        RatedMovies = self.ratings.set_index('userId').loc[UserId]['movieId']
        return RatedMovies
    
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
        UserMetrics = {'hit5 Count':hit5Count, 
                       'hit10 Count':hit10Count, 
                       'Count of Ratings': UserTestSetCount,
                       'recall5': recall5,
                       'recall10': recall10}
        
        return UserMetrics
        
    def EvaluateRecommenderModel(self):
        # Create a list to store all User Metrics
        UserMetrics = []
        
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
                        'Recall10':ModelRecall10}
        
        return ModelMetrics,EvaluationResultsforUser
                                     
        
        
            
        
        
    
    
        
        