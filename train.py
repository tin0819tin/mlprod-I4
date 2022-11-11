import os
import random
# from symbol import testlist
import pandas as pd

from tqdm import tqdm
from surprise import BaselineOnly, accuracy, Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split,GridSearchCV
from surprise import dump
from utils import * 

import requests
from itertools import groupby
import itertools
import operator
import json



class Trainer:
    def __init__(self, movie_data) -> None:
        self.movie_data = movie_data
        self.df = pd.read_csv(self.movie_data)

        # Check length of dataframe (must be more than 0)
        # Check if the file is present in the path 


    def read_process(self):
        df = self.df
        df = df.drop(columns=['Unnamed: 0', 'event', 'request', 'movie_year', 'timestamp'])
        df = df.rename(columns={'movie_rating':'ratings', 'movie_name':'item','user_id':'user'})
        reader = Reader(line_format="user item rating", sep=",")
        data = Dataset.load_from_df(df, reader=reader)
        raw_ratings = data.raw_ratings
        threshold = int(0.9 * len(raw_ratings))
        A_raw_ratings = raw_ratings[:threshold]
        B_raw_ratings = raw_ratings[threshold:]
        data.raw_ratings = A_raw_ratings
        return A_raw_ratings, B_raw_ratings, data, df

        ## checkung number of arguments 
        ## Check for data length, types

    def grid_search(self, data):
        param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005]}
        grid_search = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3)
        grid_search.fit(data)
        algo = grid_search.best_estimator["rmse"]
        return algo

        ## 
    
    def train_model(self, data, algo, ratings):
        trainset = data.build_full_trainset()
        algo.fit(trainset)
        predictions = algo.test(trainset.build_testset())
        testset = data.construct_testset(ratings)  # testset is now the set B
        return algo


if __name__ == "__main__":

    # Read Data
    movie_path = "./movie_log_10k.csv"

    # Initialize model
    team11_model = Trainer(movie_path)

    # Read data and preprocess it 
    A_raw_ratings, B_raw_ratings, data, df = team11_model.read_process()

    # Grid search to get the best model
    Training_algoritm = team11_model.grid_search(data)

    # Train model
    Algorithm = team11_model.train_model(data, Training_algoritm, B_raw_ratings)

    u = utils()
    model_name = u.serialize_model(Algorithm)

    print(model_name)

    # Users and Data definitions
    # users = df.user.unique()
    # movies = df.item.unique()

    # # Perform predictions
    # for user in tqdm(users[:5]):
    #     top5 = team11_model.get_top_n(user=user,movies=movies, algo=Algorithm, n=5)
    #     print(top5)
    


    









        
