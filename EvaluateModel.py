# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 06:29:17 2020

@author: mashimpi
"""

# Import Custom Library MovieData (in MovieData.py)
from MovieData import MovieData
from CFModel import CFModel
from MFModel import MFModel
from PopularityModel import PopularityModel
from ContentModel import ContentModel
from EvaluationMetrics import EvaluationMetrics
from sklearn.model_selection import train_test_split

# Define Global Constants
RANDOM_SEED = 100 # Setting some constant Seed Value for Sampling

# STEP 1 - Load DataSets
print('STEP 1 - Loading All DataSets')
movieData = MovieData()
movies, ratings, tags, data = movieData.LoadDataSets()
rankings = movieData.GenerateMovieStats(data)
ratingsTrain,ratingsTest = train_test_split(ratings,\
                                            stratify = ratings['userId'],\
                                            test_size = 0.2,\
                                            random_state = RANDOM_SEED)  


#cm_em = EvaluationMetrics(movies,ratingsTrain,ratingsTest,cm)

#print('Evaluating Content-Based Filtering model...')
#cb_global_metrics, cb_detailed_results_df = cm_em.EvaluateRecommenderModel()
#print('\nGlobal metrics:\n%s' % cb_global_metrics)
#print(cb_detailed_results_df.head(10))
    

#print('Top 10 Charts ...')
#print(rankings.head(10))


#Try out for an identified user
SampleUserID = 570
UserRatedMovies = movieData.GetUserTopRatings(SampleUserID,data)
print('.......TOP Movies Rated by user',SampleUserID,'..........')
print(UserRatedMovies.head(20))
pm = PopularityModel(movies,ratingsTrain,rankings)  
print('.......TOP Recommendations for POPULARITY MODEL ..........')
print(pm.RecommendMovies(SampleUserID)[['movieId','title','PredictedRating']])
pm_em = EvaluationMetrics(movies,ratings,ratingsTest,pm)
pm_global_metrics, pm_detailed_results_df = pm_em.EvaluateRecommenderModel()
print('\nGlobal metrics:\n%s' % pm_global_metrics)
print(pm_detailed_results_df.head(10))
#print('.......TOP User Keywords for CONTENT MODELLING ..........')
#print(cm.GetUserTaste(SampleUserID))
  
#cm = ContentModel(movies,ratingsTrain)
#print('.......TOP CONTENT recommendations ..........')
#print(cm.RecommendMovies(SampleUserID)[['title','genres','similarityScore']].head(20))
#cm_em = EvaluationMetrics(movies,ratings,ratingsTest, cm)
#cm_global_metrics, cm_detailed_results_df = cm_em.EvaluateRecommenderModel()
#print('\nGlobal metrics:\n%s' % cm_global_metrics)
#print('....... THE END ..........')

#uucf = CFModel(movies,ratings,'User')
#iicf = CFModel(movies,ratings,'Item')
#print("---Generating User-Movie Matrix----")
#UMM = uucf.GenerateUserMovieMatrix()

#print("---Generating Similarity Matrix----")
#Sim = uucf.GenerateSimilarityMatrix(UMM)

#print("---Generating PREDICTIONS ----")
#PredictedRatings = uucf.PredictUserRating(SampleUserID)
#print(PredictedRatings.head())
#UserRecommendations_UU = uucf.RecommendMovies(SampleUserID)
#UserRecommendations_II = iicf.RecommendMovies(SampleUserID)
#print("---------User-User TOP 20 Recommendations-----------")
#print(UserRecommendations_UU[['title']].head(20))
#print("\n\n---------Item-Item TOP 20 Recommendations-----------")
#print(UserRecommendations_II[['title']].head(20))
#print(uucf.GetUserRatedMovies(SampleUserID))
#print(Sim.set_index('userId').loc[SampleUserID].head(10))

#mf = MFModel(movies,ratings)
#UserRecommendations_SVD = mf.RecommendMovies(SampleUserID)
#print(UserRecommendations_SVD[['title']].head(20))




#print(cm.lstMovieIds)
