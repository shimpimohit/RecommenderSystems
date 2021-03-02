# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:20:44 2021

@author: mashimpi
"""

import pandas as pd
import numpy as np
from CFModel import CFModel
from MFModel import MFModel
from PopularityModel import PopularityModel
from ContentModel import ContentModel

class MFModel:
    # Define Globals
    MODEL = 'Consensus-Based'
    TOP_N = 1000
    
    def __init__(self, movies, ratings, contentModel, collabModel, mfModel, \
                 ensembleWeightContentModel, ensembleWeightCollabModel, \
                 ensembleWeightMFModel):
        self.movies = movies
        self.ratings = ratings.set_index('userId')
