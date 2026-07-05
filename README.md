# Movie-Recommendation-System-using-Machine-Learning-with-Python
Movie Recommendation System using Machine Learning with Python

Movie Recommendation System using Machine Learning with Python
A content-based movie recommendation system built with Python. The project recommends movies similar to a user's favorite movie by comparing movie metadata such as genres, keywords, tagline, cast, and director.
Project Overview
This notebook-based project uses natural language processing and cosine similarity to find movies with similar content features. The dataset contains 4,803 movies and 24 columns, including title, genres, keywords, cast, crew, director, vote information, runtime, and release details.
Features
Loads and preprocesses movie metadata from a CSV dataset
Handles missing values in selected text-based features
Combines important movie attributes into one content profile
Converts text features into numerical vectors using TF-IDF
Calculates movie similarity using cosine similarity
Accepts a user-entered movie name
Uses close string matching to handle approximate movie titles
Returns a ranked list of recommended movies
Technologies Used
Python
NumPy
Pandas
scikit-learn
TF-IDF Vectorization
Cosine Similarity
difflib
Jupyter Notebook / Google Colab
Project Structure
```text
Movie-Recommendation-System-using-Machine-Learning-with-Python-main/
├── Data/
│   ├── movies.csv
│   └── README.md
├── Movie_Recommendation_System_using_Machine_Learning_with_Python.ipynb
└── README.md
```
Dataset
The dataset is stored in:
```text
Data/movies.csv
```
Important columns used for recommendation:
`genres`
`keywords`
`tagline`
`cast`
`director`
`title`
`index`
These features are combined to create a content profile for each movie.
How It Works
Load the movie dataset using Pandas.
Select relevant features for recommendation.
Replace missing values with empty strings.
Combine selected text features into one column-like text representation.
Convert combined text into TF-IDF feature vectors.
Compute cosine similarity between all movie vectors.
Ask the user to enter a favorite movie.
Find the closest matching movie title in the dataset.
Sort movies by similarity score.
Display the top recommended movies.
Installation
Clone the repository:
```bash
git clone https://github.com/codewithsani/Movie-Recommendation-System-using-Machine-Learning-with-Python.git
cd Movie-Recommendation-System-using-Machine-Learning-with-Python
```
Install the required Python packages:
```bash
pip install numpy pandas scikit-learn notebook
```
Usage
Open the notebook:
```bash
jupyter notebook Movie_Recommendation_System_using_Machine_Learning_with_Python.ipynb
```
Run all cells in order. When prompted, enter a movie name:
```text
Enter your favourite movie name : Avatar
```
The system will display a list of recommended movies similar to the entered title.
Google Colab Usage
You can also run the notebook in Google Colab. Upload `movies.csv` to your Colab environment or update the file path in the notebook.
In the notebook, the dataset is loaded with:
```python
movies_data = pd.read_csv('/content/movies.csv')
```
For local usage, update this path to:
```python
movies_data = pd.read_csv('Data/movies.csv')
```
Example Output
```text
Movies suggested for you:

1 . Avatar
2 . Aliens
3 . Guardians of the Galaxy
4 . Star Trek Into Darkness
5 . Star Wars: Clone Wars: Volume 1
...
```
Actual recommendations may vary depending on the input movie and dataset contents.
Key Code Snippet
```python
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)
```
Future Improvements
Build a Streamlit or Flask web app interface
Add collaborative filtering
Include movie posters and descriptions
Add filters for genre, rating, release year, and language
Improve ranking using popularity and vote average
Save the similarity matrix for faster recommendations
License
This project is for educational purposes. Add a license file if you plan to distribute or reuse it publicly.
Author
Saied Sani
