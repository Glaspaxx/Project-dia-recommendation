# Import necessary libraries
import os
import re
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def download_files_to_directory(url,data_folder,output_directory):
    zip_name = url.split('/')[::-1][0]

    # We create the data and output folders if they don't exist to organize our work
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Downloading the dataset from the web using urllib.request
    urllib.request.urlretrieve(url, os.path.join(data_folder, zip_name))

    # Unziping all the files to work with them and removing the original zip file
    with zipfile.ZipFile(os.path.join(data_folder, zip_name), 'r') as zip_ref:
        zip_ref.extractall(data_folder)

    if os.path.exists(os.path.join(data_folder, zip_name)):
        os.remove(os.path.join(data_folder, zip_name))
    
url = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip' 
#uncomment the line bellow the first time running the code to download the dataset   
#download_files_to_directory(url,'data','output')

# After downloading the dataset and unziping it let start by loading the MovieLens dataset
ratings = pd.read_csv(f'data/{url.split("/")[::-1][0].replace(".zip","")}/ratings.csv')
movies = pd.read_csv(f'data/{url.split("/")[::-1][0].replace(".zip","")}/movies.csv')
links = pd.read_csv(f'data/{url.split("/")[::-1][0].replace(".zip","")}/links.csv')

# cleaning the titles of the movies to facilitate the search in the dataset
def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)#replaces anything different than uppercase letters, lowercase letters and digits using REGEX
    return title
movies["clean_title"] = movies["title"].apply(clean_title)

# we will use tfidf calculation for our search function to be more like a search engine
vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()# compare our query to each of the clean titles, return how similar it is
    indices = np.argpartition(similarity, -5)[-5:]# the 5 titles that have the most similarty to our seach term
    results = movies.iloc[indices].iloc[::-1] # reversing the array, the most similar result is the last in the array
    return results

def creating_links(imdbId):
    imdbId_str = str(imdbId)
    if len(imdbId_str) < 7:
        imdbId_str = "0" * (7 - len(imdbId_str)) + imdbId_str
    
    imdb_link = f"https://www.imdb.com/title/tt{imdbId_str}/"
    return imdb_link

# Train a filtering model
def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()# liked the same movie as movie_id and gave a rating greater than 4, and find their unique user id 
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]# find other movies that they liked and gave it a rating more than 4 and find the movie ids of these movies
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)# how many times a movie appears / on the number of users

    similar_user_recs = similar_user_recs[similar_user_recs > .10]# selecting only the movies that 10 percent or more users similar to us liked 
    # we will check our results with users that are not in our similar_users array 
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]# how much all of our users in dataset liked these movies
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())# find the percentage of all users that recommended the movies that users like us liked
    # we need to find a score to judge the movies we will recommend
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    # similar is users that liked the same movie as us, all is the all users in the dataset how much they liked the same movie
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)# sorting our scores descending, the higher the score the better the recommendation is 
    recommended_movies_links = rec_percentages.merge(movies, left_index=True, right_on="movieId") # merge our rec_percentages array with the movies dataset to get the title of the movies
    recommended_movies_links = recommended_movies_links.merge(links, how='left', on="movieId")# merge our rec_percentages array with the links dataset to get the links of the movies
    recommended_movies_links["imdbId_links"] = recommended_movies_links["imdbId"].apply(creating_links)
    return recommended_movies_links[["movieId","score", "title", "genres","imdbId_links"]].head(10)


def find_movies(movie_name):
    results = search(movie_name)
    movie_id = results.iloc[0]["movieId"]
    return find_similar_movies(movie_id)

# example
movie_name = input("Type the name of the movie you watched : \n")
print(find_movies(movie_name))
