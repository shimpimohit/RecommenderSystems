# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:03:00 2020

@author: mashimpi
"""

# Important Libraries
import pandas as pd

class MovieData:
    
    userRatingsPath = '../Data/ml-latest-small/ratings.csv'
    moviesPath = '../Data/ml-latest-small/movies.csv'
    tagsPath = '../Data/ml-latest-small/tags.csv'
    metadataPath = '../Data/ml-latest-small/movies_metadata.csv'
    linksPath = '../Data/ml-latest-small/links.csv'
    
    # Load DataSets
    def LoadUserRatings(self):
        print("Loading Ratings DataSet ...")
        ratings = pd.read_csv(self.userRatingsPath)
        # Housekeeping - Sort Ratings by MovieId
        ratings.sort_values(by='movieId', inplace=True)
        ratings.reset_index(inplace=True, drop=True)
        # Housekeeping - Convert Timestamp to readable format
        #ratings['lastUpdateDate'] = pd.to_datetime(ratings.timestamp, infer_datetime_format=True)
        ratings['lastUpdateDate'] = pd.to_datetime(ratings.timestamp, \
                                                   unit = 's', \
                                                   errors = 'coerce')
        
        return ratings
    
    def LoadMovies(self):
        print("Loading Movies DataSet ...")
        movies = pd.read_csv(self.moviesPath)
        links = pd.read_csv(self.linksPath)
        metadata = pd.read_csv(self.metadataPath)
        
        # Join MOVIES, LINKS and METADATA Dataframes for getting movie attrubutes
        metadata = metadata[['id','imdb_id','original_title','overview',
                             'popularity','release_date','runtime',
                             'vote_average','vote_count']]
        metadata['id'] = metadata['id'].astype(int)
        movies = pd.merge(movies, links, how='left',left_on='movieId',right_on='movieId')
        movies = pd.merge(movies,metadata, how='left',left_on='tmdbId',right_on='id')
        
        # Housekeeping - Sorting movies by MovieID
        movies.sort_values(by='movieId', inplace=True)
        movies.reset_index(inplace=True, drop=True)
        
        # Housekeeping - splitting YEAR from TITLE
        movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
        movies.year = pd.to_datetime(movies.year, format='%Y')
        movies.year = movies.year.dt.year
        movies['movieName'] = movies.title.str[:-7]
        
        # Housekeeping - Splitting GENRES into columns
        movies = movies.join(movies.genres.str.get_dummies().astype(bool))
        #movies.drop('genres', inplace=True, axis=1)
        #genres_df = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
        #genres_df = pd.DataFrame(genres_df, columns=['genre'])
        
        # ..... Data Cleansing .......
        # overview column will be used for CONTENT-BASED Model. Hence clensing
        movies['overview'] = movies['overview'].fillna('')
        return movies
    
    def LoadTags(self):
        print("Loading Tags DataSet ...")
        return pd.read_csv(self.tagsPath)
    
    def LoadDataSets(self):
        # This function loads all data and also pre-processes the data
        # Pre-Processing Includes:
        #     a. <TBD>
        
        ratings = self.LoadUserRatings()
        print("Loaded",ratings.shape[0],"user ratings ...\n")
                
        movies = self.LoadMovies()
        print("Loaded",movies.shape[0],"movies ...\n")
        
        tags = self.LoadTags()
        print("Loaded",tags.shape[0],"tags ...\n")
        
        print("Enriching Ratings Dataset with Movie Information ...\n")
        data = ratings.merge(movies,on='movieId',how='left')
        
        print("Datasets Loaded.")
        return (movies, ratings, tags, data)
    
    def GenerateMovieStats(self,data):
        print("Generating Movie Statistics ...")
        
        # For Weighted Ratings, we define the PERCENTILE CUT-OFF in Vote Count   
        # For sake of simplicity, this is hardcoded rather than parameterised
        QUANTILE_THRESHOLD = 0.85
                
        movie_stats_df = data[['movieId','title']].drop_duplicates()
        movie_stats_df.set_index(['movieId'],inplace = True)
        movie_stats_df['Rating_Mean'] = data.groupby('movieId')['rating'].mean()
        movie_stats_df['Rating_Count'] = data.groupby('movieId')['rating'].count()
        
        # Lets Calculate Weighted Ratings
        # Weighted_Rating = (v / (v + m)) * R + (m / (v + m)) * C
        # where,
        #       v = Rating_Count or number of ratings each movie recieved
        #       m = minimum number of ratings required to qualify as popular
        #       R = Rating_Mean or Mean Rating for the movie
        #       C = Mean_Rating for all movies in the dataset
        
        C = data.rating.mean()
        m = movie_stats_df['Rating_Count'].quantile(QUANTILE_THRESHOLD)
        
        movie_stats_df['Weighted_Rating'] = (movie_stats_df['Rating_Mean'] * 
                                             movie_stats_df['Rating_Count'] / 
                                             (m + movie_stats_df['Rating_Count'])) \
                                           + (m * C / (m + movie_stats_df['Rating_Count']))
                                             
        movie_stats_df = movie_stats_df.sort_values('Weighted_Rating', ascending=False)
        return movie_stats_df
    
    #General Funtions
    def GetUserRatedMovies(self, UserId, ratings):
        RatedMovies = ratings.set_index('userId').loc[UserId]['movieId']
        return RatedMovies
    
    # Function to see the TOP Movies as Rated by particular user
    def GetUserTopRatings(self, UserId,data):
        TopMoviesForUser = data.set_index('userId').loc[UserId]\
                               .sort_values('rating',ascending = False)\
                                   [['movieId','title']]
                                   #[['title','genres','rating']]
        return TopMoviesForUser
    
    def getUserRatings(self, userID):
        return 1









