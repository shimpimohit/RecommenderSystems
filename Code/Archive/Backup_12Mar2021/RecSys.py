# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 06:29:17 2020

@author: mashimpi
"""

# Import Custom Library MovieData (in MovieData.py)
from MovieData import MovieData
from EvaluateModel import EvaluateModel
from CFModel import CFModel
from MFModel import MFModel
from PopularityModel import PopularityModel
from ContentModel import ContentModel
from sklearn.model_selection import train_test_split

# Define Global Constants
RANDOM_SEED = 100 # Setting some constant Seed Value for Sampling

# STEP 1 - Load DataSets
print('STEP 1 - Loading All DataSets')
print('---------------------------------------------------------------------')
movieData = MovieData()
movies, ratings, tags, data = movieData.LoadDataSets()
rankings = movieData.GenerateMovieStats(data)
ratingsAntiTest,ratingsTest = train_test_split(ratings,\
                                               stratify = ratings['userId'],\
                                               test_size = 0.2,\
                                               random_state = RANDOM_SEED)

print('STEP 2 - Constructing Recommendations Models')
print('---------------------------------------------------------------------')
#cf = EvaluateModel(movies,ratings,'CFModel')

print('STEP 3 - Evaluating Recommendations Models')
print('---------------------------------------------------------------------')
evalCM = EvaluateModel(movies,ratings,'ContentModel')
CMModelMetrics,CMEvaluationResultsforUser = evalCM.EvaluateRecommenderModel()
    

    
    #trainUserMovieMatrix = np.zeros(len(ratings.userId.unique()),\
#                                len(ratings.movieId.unique()))    


#cfmk30_em = EvaluationMetrics(movies,ratingsTrain,ratingsTest, cfmk30)
#cfmk30_global_metrics, cfmk30_detailed_results_df = cfmk30_em.EvaluateRecommenderModel()


#cm_em = EvaluationMetrics(movies,ratingsTrain,ratingsTest,cm)

#print('Evaluating Content-Based Filtering model...')
#cb_global_metrics, cb_detailed_results_df = cm_em.EvaluateRecommenderModel()
#print('\nGlobal metrics:\n%s' % cb_global_metrics)
#print(cb_detailed_results_df.head(10))
    

#print('Top 10 Charts ...')
#print(rankings.head(10))


#Try out for an identified user
#SampleUserID = 570
#UserRatedMovies = movieData.GetUserTopRatings(SampleUserID,data)
#print('.......TOP Movies Rated by user',SampleUserID,'..........')
#print(UserRatedMovies.head(20))
#ubcf = CFModel(movies,ratingsTrain,'user')
#ibcf = CFModel(movies,ratingsTrain,'item')
#ii = CFModel(movies,ratings,'item')
#ii2 = CFModel(movies,ratings,'item',2)
#ubcf = CFModel(movies,ratingsTrain,'User')
#ubcf2 = CFModel(movies,ratingsTrain,'User',2)
#print("---Generating PREDICTIONS ----")
#PredictedRatings = ubcf.PredictUserRating(SampleUserID)
#PredictedRatings = ubcf.PredictRating()
#PredictedRating570 = ubcf.PredictedRatings.loc[570]
#Recommendations570 = ubcf.RecommendMovies(570)
#UserRecommendations = ubcf.PredictRating()
#print(UserRecommendations.head(20))
#R570i = ibcf.RecommendMovies(SampleUserID)
#print(R570u[['title','PredictedRating']].head(20))
#print(R570i[['title','PredictedRating']].head(20))

#ubcf_em = EvaluationMetrics(movies,ratings,ratingsTest, ubcf)
#ubcf_global_metrics, ubcf_detailed_results_df = ubcf_em.EvaluateRecommenderModel()
#print('\nGlobal UBCF metrics:\n%s' % ubcf_global_metrics)
#ibcf_em = EvaluationMetrics(movies,ratings,ratingsTest, ibcf)
#ibcf_global_metrics, ibcf_detailed_results_df = ibcf_em.EvaluateRecommenderModel()
#print('\nGlobal IBCF metrics:\n%s' % ubcf_global_metrics)
#print('....... THE END ..........')


#pm = PopularityModel(movies,ratingsTrain,rankings)  
#print('.......TOP Recommendations for POPULARITY MODEL ..........')
#print(pm.RecommendMovies(SampleUserID)[['movieId','title','PredictedRating']])
#pm_em = EvaluationMetrics(movies,ratings,ratingsTest,pm)
#pm_global_metrics, pm_detailed_results_df = pm_em.EvaluateRecommenderModel()
#print('\nGlobal metrics:\n%s' % pm_global_metrics)
#print(pm_detailed_results_df.head(10))
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
