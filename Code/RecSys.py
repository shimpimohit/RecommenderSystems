# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 06:29:17 2020

@author: mashimpi
"""

# Import Custom Library MovieData (in MovieData.py)
from MovieData import MovieData
from EvaluateModel import EvaluateModel
from CFModel import CFModel
from CustomMF import CustomMF
#from MFModel import MFModel
from PopularityModel import PopularityModel
from ContentModel import ContentModel
import matplotlib.pyplot as plt
import pandas as pd


# Define Global Constants
RANDOM_SEED = 100 # Setting some constant Seed Value for Sampling
SAMPLE_USER = 570

# STEP 1 - Load DataSets
print('\nSTEP 1 - Loading All DataSets')
print('---------------------------------------------------------------------')
dataset = MovieData()


print('\nSTEP 2 - Constructing Recommendations Models')
print('---------------------------------------------------------------------')
print('# 1 ---------- Collaborative Filtering Models ------------------------')
print('\n... 1.1 Memory Based Collaborative Filtering (User Based)')
CFUB_Cosine_NoColdStart = CFModel(movieData = dataset,
                                  CFMethod = 'User',
                                  Metric = 'Cosine',
                                  EliminateBias = True,
                                  modelMode = 'Train')
CFUB_Cosine_WithColdStart = CFModel(movieData = dataset,
                                    CFMethod = 'User',
                                    Metric = 'Cosine',
                                    EliminateBias = False,
                                    modelMode = 'Train')
CFUB_Pearson_NoColdStart = CFModel(movieData = dataset,
                                   CFMethod = 'User',
                                   Metric = 'Pearson',
                                   EliminateBias = True,
                                   modelMode = 'Train')
CFUB_Pearson_WithColdStart = CFModel(movieData = dataset,
                                     CFMethod = 'User',
                                     Metric = 'Pearson',
                                     EliminateBias = False,
                                     modelMode = 'Train')
print('\n... 1.2 Memory Based Collaborative Filtering (Item Based)')
CFIB_Cosine_NoColdStart = CFModel(movieData = dataset,
                                  CFMethod = 'User',
                                  Metric = 'Item',
                                  EliminateBias = True,
                                  modelMode = 'Train')
CFIB_Cosine_WithColdStart = CFModel(movieData = dataset,
                                    CFMethod = 'Item',
                                    Metric = 'Cosine',
                                    EliminateBias = False,
                                    modelMode = 'Train')
CFIB_Pearson_NoColdStart = CFModel(movieData = dataset,
                                   CFMethod = 'Item',
                                   Metric = 'Pearson',
                                   EliminateBias = True,
                                   modelMode = 'Train')
CFIB_Pearson_WithColdStart = CFModel(movieData = dataset,
                                     CFMethod = 'Item',
                                     Metric = 'Pearson',
                                     EliminateBias = False,
                                     modelMode = 'Train')

print('# 2 ---------- Matrix Factorization Models ---------------------------')
print('\n... 2.1 Matrix Factorization with SGD')
MF_SGD = CustomMF(movieData = dataset, OptimizationMethod='SGD',
                  modelMode = 'Train')
print('\n... 2.1 Matrix Factorization with ALS')
MF_ALS = CustomMF(movieData = dataset, OptimizationMethod='ALS',
                  modelMode = 'Train')

print('# 3 ---------- Content Based Model -----------------------------------')
print('\n... 3.1 Content Based Model with User-Taste Profile')
CM = ContentModel(dataset)

print('# 4 ---------- Popularity Model --------------------------------------')
print('\n... 4.1 Personalized Popularity Model')
PM = PopularityModel(dataset)


print('\nSTEP 3 - Evaluating Recommendations Models')
print('----------------------------------------------------------------------')
# 1 - Collaborative Filtering
print('# 1 ---------- Evaluating Collaborative Filtering Models -------------')
print('\n... 1.1 Evaluating Memory Based Collaborative Filtering (User Based)')

print('\n... Evaluating Memory Based CF (User Based) for Cosine Similarity without Cold Start')
eval_CFUB_Cosine_NoColdStart = EvaluateModel(CFUB_Cosine_NoColdStart)
CFUB_Cosine_NoColdStart_Metrics,CFUB_Cosine_NoColdStart_Results \
    = eval_CFUB_Cosine_NoColdStart.EvaluateRecommenderModel()

print('\n... Evaluating Memory Based CF (User Based) for Cosine Similarity with Cold Start')
eval_CFUB_Cosine_WithColdStart = EvaluateModel(CFUB_Cosine_WithColdStart)
CFUB_Cosine_WithColdStart_Metrics,CFUB_Cosine_WithColdStart_Results \
    = eval_CFUB_Cosine_WithColdStart.EvaluateRecommenderModel()    
    
print('\n... Evaluating Memory Based CF (User Based) for Pearson Similarity without Cold Start')
eval_CFUB_Pearson_NoColdStart = EvaluateModel(CFUB_Pearson_NoColdStart)
CFUB_Pearson_NoColdStart_Metrics,CFUB_Pearson_NoColdStart_Results \
    = eval_CFUB_Pearson_NoColdStart.EvaluateRecommenderModel()    

print('\n... Evaluating Memory Based CF (User Based) for Pearson Similarity with Cold Start')
eval_CFUB_Pearson_WithColdStart = EvaluateModel(CFUB_Pearson_WithColdStart)
CFUB_Pearson_WithColdStart_Metrics,CFUB_Pearson_WithColdStart_Results \
    = eval_CFUB_Pearson_WithColdStart.EvaluateRecommenderModel()

print('\n... 1.2 Evaluating Memory Based Collaborative Filtering (Item Based)')
print('\n... Evaluating Memory Based CF (Item Based) for Cosine Similarity without Cold Start')
eval_CFIB_Cosine_NoColdStart = EvaluateModel(CFIB_Cosine_NoColdStart)
CFIB_Cosine_NoColdStart_Metrics,CFIB_Cosine_NoColdStart_Results \
    = eval_CFIB_Cosine_NoColdStart.EvaluateRecommenderModel()

print('\n... Evaluating Memory Based CF (Item Based) for Cosine Similarity with Cold Start')
eval_CFIB_Cosine_WithColdStart = EvaluateModel(CFIB_Cosine_WithColdStart)
CFIB_Cosine_WithColdStart_Metrics,CFIB_Cosine_WithColdStart_Results \
    = eval_CFIB_Cosine_WithColdStart.EvaluateRecommenderModel()    
    
print('\n... Evaluating Memory Based CF (Item Based) for Pearson Similarity without Cold Start')
eval_CFIB_Pearson_NoColdStart = EvaluateModel(CFIB_Pearson_NoColdStart)
CFIB_Pearson_NoColdStart_Metrics,CFIB_Pearson_NoColdStart_Results \
    = eval_CFIB_Pearson_NoColdStart.EvaluateRecommenderModel()    

print('\n... Evaluating Memory Based CF (Item Based) for Pearson Similarity with Cold Start')
eval_CFIB_Pearson_WithColdStart = EvaluateModel(CFIB_Pearson_WithColdStart)
CFIB_Pearson_WithColdStart_Metrics,CFIB_Pearson_WithColdStart_Results \
    = eval_CFIB_Pearson_WithColdStart.EvaluateRecommenderModel()

# 2 - Matrix Factorization Models
print('# 2 ---------- Evaluating Matrix Factorization Models ----------------')
print('\n... 2.1 Evaluating Matrix Factorization with SGD')
eval_MF_SGD = EvaluateModel(MF_SGD)
MF_SGD_Metrics,MF_SGD_Results = eval_MF_SGD.EvaluateRecommenderModel()

print('\n... 2.2 Evaluating Matrix Factorization with ALS')
eval_MF_ALS = EvaluateModel(MF_ALS)
MF_ALS_Metrics,MF_ALS_Results = eval_MF_ALS.EvaluateRecommenderModel()

# 3 - Content Based Model
print('# 3 ---------- Evaluating Content Based Model ------------------------')
print('\n... 3.1 Evaluating Content Based Model with User-Taste Profile')
eval_CM = EvaluateModel(CM)
CM_Metrics,CM_Results = eval_CM.EvaluateRecommenderModel()

# 4 - Popularity Model
print('# 4 ---------- Evaluating Popularity Model ---------------------------')
print('\n... 4.1 Evaluating Personalized Popularity Model')
eval_PM = EvaluateModel(PM)
PM_Metrics,PM_Results = eval_PM.EvaluateRecommenderModel()

# -----------------------------PLOT CHARTS------------------------------------
AllModelMetrics = pd.DataFrame([PM_Metrics, CM_Metrics, MF_ALS_Metrics, \
                                MF_SGD_Metrics,CFIB_Pearson_WithColdStart_Metrics,\
                                CFIB_Pearson_NoColdStart_Metrics,\
                                CFIB_Cosine_WithColdStart_Metrics,\
                                CFIB_Cosine_NoColdStart_Metrics,\
                                CFUB_Cosine_NoColdStart_Metrics,\
                                CFUB_Cosine_WithColdStart_Metrics,\
                                CFUB_Pearson_NoColdStart_Metrics,\
                                CFUB_Pearson_WithColdStart_Metrics]) \
                        .set_index('ModelName')

# Plot the MSE

# Plot the Recall 5 and Recall 10 Metrics
AllModelMetrics.reset_index().sort_values(by = 'Recall5').plot\
    .barh(x="ModelName",y=["Recall5","Recall10"],title="Recall by Model")
    
AllModelMetrics.reset_index().sort_values(by = 'MSE').plot\
    .barh(x="ModelName",y=["MSE","MAE"],title="MSE & MAE by Model")    