import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

#variables being used 
user_movie = "Avatar"
cv = TfidfVectorizer() #CountVectorizer()
movie_data = pd.read_csv('movie.csv')
#processing 
movie_features = movie_data[['keywords','cast','genres','director', 'tagline']]
#merging into one column 
movie_data['combined'] = movie_data.apply(lambda row: ' '.join(row.astype(str)), axis=1)

#get the count and cosine sim
matrix = cv.fit_transform(movie_data['combined'])
similarity = cosine_similarity(matrix)


#functions for processing

    #implementing popularity into ranking: using hybrid scoring 
    #weighted average -> multiply each score by weight%, add products together, divide by sum of weights 
    #weighted ratings -> multiply number of votes by average rating of movie + divide by sum of weights -> multiply minimum votes required to be considered + divide by sum of weights 
mean_vote_data = movie_data['vote_average'].mean()
min_votes = movie_data['vote_count'].quantile(0.8)
def weighted_rating(x, mean_vote_data=mean_vote_data, min_votes=min_votes):
    num_votes = x['vote_count']
    avg_rating = x['vote_average']
    
    return (num_votes/(num_votes+min_votes))*avg_rating + (min_votes/(num_votes+min_votes))*mean_vote_data 

def get_recommended(movie_index):
    alpha = 0.7 #weight for similarity 
    beta = 0.3  #weight for popularity 
    movie_data_copy = movie_data.copy()
    movie_data_copy['popularity_score'] = movie_data_copy.apply(weighted_rating, axis=1)

    if movie_data_copy['popularity_score'].max() != movie_data_copy['popularity_score'].min():
        movie_data_copy['popularity_score'] = (movie_data_copy['popularity_score']-movie_data_copy['popularity_score'].min())/(movie_data_copy['popularity_score'].max()-movie_data_copy['popularity_score'].min())
    else:
        movie_data_copy['popularity_score'] = 0

    movie_data_copy['similarity'] = similarity[movie_index]
    if movie_data_copy['similarity'].max() != movie_data_copy['similarity'].min():
        movie_data_copy['similarity'] = (movie_data_copy['similarity']-movie_data_copy['similarity'].min())/(movie_data_copy['similarity'].max()-movie_data_copy['similarity'].min())
    else:
        movie_data_copy['similarity']=0

    
    #hybrid score 
    movie_data_copy['hybrid'] = alpha * movie_data_copy['similarity'] + beta * movie_data_copy['popularity_score']

    recommended = movie_data_copy[movie_data_copy.index != movie_index].sort_values('hybrid',ascending=False)
    return recommended['title'].head(10)

#getting input from user in (back-end)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    user_movie = request.form.get('user_movie')
    user_movie_copy = user_movie.strip().lower()
    title_matches = movie_data[movie_data['title'].str.lower().str.strip()==user_movie_copy]
    if title_matches.empty:
        return render_template('index.html', movie=user_movie, recommended=["Movie not found."])
    else:
        movie_index = title_matches.iloc[0].name
        similar_movies = list(enumerate(similarity[movie_index])) # gives the list of tuples
        sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse=True)
    
        recommended_df = get_recommended(movie_index).tolist()
        recommended = recommended_df

        return render_template('index.html', movie=user_movie, recommended=recommended)


if __name__=='__main__':
    app.run(debug=True)

#print("\n".join(recommended['title'].head(10)))


# what other additional features can be used to improve the quality of the recommendation system