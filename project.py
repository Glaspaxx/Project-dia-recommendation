# Import necessary libraries
import os
import re
import urllib.request
import requests
import zipfile
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import streamlit as st
import warnings; warnings.simplefilter('ignore')

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
    
url_1 = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip' 
url_2 = ''
#uncomment the line bellow the first time running the code to download the dataset   
#download_files_to_directory(url,'data','output')

# After downloading the dataset and unziping it let start by loading the MovieLens dataset
ratings = pd.read_csv(f'data/{url_1.split("/")[::-1][0].replace(".zip","")}/ratings.csv')
movies = pd.read_csv(f'data/{url_1.split("/")[::-1][0].replace(".zip","")}/movies.csv')
links = pd.read_csv(f'data/{url_1.split("/")[::-1][0].replace(".zip","")}/links.csv')

# second dataset
df1=pd.read_csv('data/archive/tmdb_5000_credits.csv')
df2=pd.read_csv('data/archive/tmdb_5000_movies.csv')

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

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500" + poster_path
    return full_path

# content based filtering
def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] >= 4)]["userId"].unique()# liked the same movie as movie_id and gave a rating greater than 4, and find their unique user id 
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] >= 4)]["movieId"]# find other movies that they liked and gave it a rating more than 4 and find the movie ids of these movies
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)# how many times a movie appears / on the number of users

    similar_user_recs = similar_user_recs[similar_user_recs > .10]# selecting only the movies that 10 percent or more users similar to us liked 
    # we will check our results with users that are not in our similar_users array 
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] >= 4)]# how much all of our users in dataset liked these movies
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
    recommended_movies_links["posters"] = recommended_movies_links["tmdbId"].apply(fetch_poster)
    return recommended_movies_links[["movieId","score", "title", "genres","imdbId_links","posters"]].head(10)


# collaborative filtering
def collaborative_filter(userId, movieId,r_ui ):
    reader = Reader()
    ratings = pd.read_csv('data/archive/ratings_small.csv')
    ratings.head()

    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    svd = SVD()
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    return svd.predict(userId,movieId,r_ui)


# Hybrid filtering
def hybrid(userId, title):
    movies = movies.merge(links, how='left', on="movieId")# merge our rec_percentages array with the links dataset to get the links of the movies
    movies["imdbId_links"] = movies["imdbId"].apply(creating_links)
    movies["posters"] = movies["tmdbId"].apply(fetch_poster)
    movies[["movieId","score", "title", "genres","imdbId_links","posters","tmdbId"]]
    idx = movies[title]
    tmdbId = movies.loc[title]['tmdbId']
    movie_id = movies.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_similarity[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = movies.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, movies.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)      

def find_movies(movie_name):
    results = search(movie_name)
    movie_id = results.iloc[0]["movieId"]
    return find_similar_movies(movie_id)

# example content based 
#movie_name = input("Type the name of the movie you watched : \n")
#print(find_movies(movie_name))

#example collaborative based
#print(collaborative_filter(1, 302, 3))

#example hybrid based
#print(hybrid(1,'Avatar'))


# web interface below
st.header('Hybrid Movie Recommender System Using Machine Learning')

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movies = find_movies(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movies["title"][1])
        st.image(recommended_movies["posters"][1])
    with col2:
        st.text(recommended_movies["title"][2])
        st.image(recommended_movies["posters"][2])

    with col3:
        st.text(recommended_movies["title"][3])
        st.image(recommended_movies["posters"][3])
    with col4:
        st.text(recommended_movies["title"][4])
        st.image(recommended_movies["posters"][4])
    with col5:
        st.text(recommended_movies["title"][5])
        st.image(recommended_movies["posters"][5])




