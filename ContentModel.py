# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 06:29:17 2020

@author: mashimpi
"""

# ------ CONTENT BASED RECOMMENDER MODEL ---------- #


import numpy as np
import pandas as pd
import scipy
#from MovieData import MovieData

#Import TfIdfVectorizer and linear_kernel from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.preprocessing

#---Code for Content Based Model----#
class ContentModel:
    
    MODEL = 'Content-Based'
    
    TOP_N = 5000
    
    def __init__(self, movies,ratings):
        self.movies = movies
        self.ratings = ratings.set_index('userId')
        self.lstMovieIds = movies['movieId'].tolist()
        
        # Create TF-IDF Vector Matrix that measures the relevance of all Tokens
        # for the Movie 'Title' and the 'Overview' String        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.TokenCorpus = self.movies['original_title'].fillna('') + " " + \
                           self.movies['overview'] #+ " " + \
                           #self.movies['genres'].fillna('').replace("|"," ")    
        self.TFIDF_Matrix = self.tfidf_vectorizer.fit_transform(self.TokenCorpus)
        self.TFIDF_Tokens = self.tfidf_vectorizer.get_feature_names()
        
    # Create a TFIDF Vector (slice of the TFIDF Matrix) for a particular Movie
    def GenerateMovieProfile_TFIDF(self,MovieId):
        idx = self.lstMovieIds.index(MovieId)
        #TFIDF_Matrix = self.GenerateMatrix_TFIDF()
        #print('Generating TFIDF Matrix for Movie',MovieId)
        MovieProfile = self.TFIDF_Matrix[idx:idx+1]
        #print('Generated TFIDF Matrix for Movie',MovieId)
        return MovieProfile
    
    # Create a slice of the TFIDF Matrix for a particular set of Movies
    def GenerateMovieProfiles_TFIDF(self,lstUserRatedMovieIds):
        lstMovieProfile = [self.GenerateMovieProfile_TFIDF(MovieId) \
                           for MovieId in lstUserRatedMovieIds]
        MovieProfiles = scipy.sparse.vstack(lstMovieProfile)
        return MovieProfiles
    
    # Create a Avg Weighted Rating Vector for a user based on all the Movies
    # rated by the user and multiplying it by the TFIDF relevance of the movie
    def GenerateUserProfile_TFIDF(self,UserId):
        # Firstly, get all the ratings that the user has given to all movies
        UserRatings = self.ratings.loc[UserId]
        
        # Generate a vector of all user ratings
        arrUserRatings = np.array(UserRatings['rating']).reshape(-1,1)
        
        # Generate the User Profile Vector by Generating the 
        UserMovieProfile = self.GenerateMovieProfiles_TFIDF(UserRatings['movieId'])
        
        # Generate the Weighted User Profile and normalize it
        UserRating_WeightedAvg = np.sum(UserMovieProfile.multiply(arrUserRatings) \
                                        , axis=0) / np.sum(arrUserRatings)
        
        UserRating_WeightedAvg = sklearn.preprocessing.normalize(UserRating_WeightedAvg)
        return UserRating_WeightedAvg
    
    # Using Above function, create User Profile for a set of users
    def GenerateAllUsersProfile_TFIDF(self):
        UserRatings_TFIDF = {}
        #ratings = self.ratings.set_index('userId')
        
        for UserId in self.ratings.index.unique():
            UserRatings_TFIDF[UserId] = self.GenerateUserProfile_TFIDF(UserId)
            #print('Generated User Profile for User ID',UserId)
            print('Processed Users = ',len(UserRatings_TFIDF))
            
        return UserRatings_TFIDF      
    
    #Function to validate User Taste
    def GetUserTaste(self,UserId,n=10):
        UserTaste = pd.DataFrame(sorted(zip(self.TFIDF_Tokens, \
                                            self.GenerateUserProfile_TFIDF(UserId).\
                                            #self.GenerateAllUsersProfile_TFIDF()[UserId].\
                                                flatten().tolist()),\
                                        key = lambda x: -x[1])[:n],\
                                 columns=['Token', 'RelevanceScore'])
        return UserTaste
    
    def GetModel(self):
        return self.MODEL
    
    def GetSimilarMoviesForUser(self,UserId):
        # Generate Similarity Matrix based on COSINE SIMILARITY
        SimilarityMatrix = cosine_similarity(self.GenerateUserProfile_TFIDF(UserId), \
                                              self.TFIDF_Matrix)
        # Get the Top Similar Movies based on the similarity Matrix
        SimilarMovieIndices = SimilarityMatrix.argsort().flatten()[(-1 * self.TOP_N):]
        SimilarMovies = sorted([ (self.lstMovieIds[IndexCounter],\
                                  SimilarityMatrix[0,IndexCounter]) \
                                     for IndexCounter in SimilarMovieIndices ], \
                                     key = lambda x: -x[1])
        return SimilarMovies
    
    # Get Already rated movies that need to be excluded from recommendations
    def GetUserRatedMovies(self, UserId):
        RatedMovies = self.ratings.loc[UserId]['movieId']
        return RatedMovies
    
    # Recommendation Function
    def RecommendMovies(self, UserId):
        # Get all the ranked similar movies
        UserRecommendations = self.GetSimilarMoviesForUser(UserId)
        
        # Exclusion List of all movies that user has already rated
        ExcludeList = self.GetUserRatedMovies(UserId)
        
        # Master List of recommendations = User Recommendations - Exclusion List
        #UserRecommendations = list(filter(lambda lstMovies: lstMovies[0] \
        #                                  not in ExcludeList, UserRecommendations))
        
        UserRecommendations = [lstMovies for lstMovies in UserRecommendations \
                               if all(lstExcluded not in lstMovies \
                                      for lstExcluded in ExcludeList)]
            
        UserRecommendations = pd.DataFrame(UserRecommendations,\
                                           columns = ['movieId','similarityScore'])\
                                           .sort_values(by='similarityScore',\
                                                        ascending = False)\
                                           .head(self.TOP_N)
                                           
        UserRecommendations = UserRecommendations.merge(self.movies,\
                                                        how = 'left', \
                                                        left_on = 'movieId',\
                                                        right_on = 'movieId')
            
        return UserRecommendations[['movieId','title','genres','similarityScore']] 
    
    
        
        
        
        
    

