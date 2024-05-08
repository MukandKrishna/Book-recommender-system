import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
import sys
import psycopg2

# Set console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Initialize connection to PostgreSQL database
conn = psycopg2.connect(
    dbname="DLP",
    user="postgres",
    password="mukand",
    host="127.0.0.1",
    port="8000"
)

# Read Candidates and Courses data from the database
books_query = """SELECT isbn, "Book-Title", "Book-Author", "Year-Of-Publication", publisher,"Image-URL-L" FROM public.books;"""
users_query = """SELECT * FROM public.users;"""
ratings_query = """SELECT * FROM public.ratings;"""

books = pd.read_sql(books_query, con=conn)
users = pd.read_sql(users_query, con=conn)
ratings = pd.read_sql(ratings_query, con=conn)


# books = pd.read_csv('Books.csv', encoding='latin-1')
# users = pd.read_csv('Users.csv', encoding='latin-1')
# ratings = pd.read_csv('Ratings.csv', encoding='latin-1')

# books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]

books.rename(columns={"Book-Title": "Title", "Book-Author": 'Author',
             'Year-Of-Publication': 'Year', 'Image-URL-L': 'Img-URL'}, inplace=True)

users.rename(columns={'User-ID': 'ID'}, inplace=True)

ratings.rename(
    columns={'User-ID': 'ID', 'Book-Rating': 'Rating'}, inplace=True)

x = ratings['ID'].value_counts() > 200
# x[x].shape

y = x[x].index

ratings = ratings[ratings['ID'].isin(y)]

ratings_with_books = ratings.merge(books, on="isbn")

num_rating = ratings_with_books.groupby(
    'Title')['Rating'].count().reset_index()

num_rating.rename(columns={'Rating': 'Num_of_rating'}, inplace=True)

final_rating = ratings_with_books.merge(num_rating, on='Title')

final_rating = final_rating[final_rating['Num_of_rating'] >= 50]

# Lets create a pivot table
book_pivot = final_rating.pivot_table(
    columns='ID', index='Title', values='Rating')

book_pivot.fillna(0, inplace=True)

book_sparse = csr_matrix(book_pivot)

model = NearestNeighbors(algorithm='brute')

model.fit(book_sparse)

distance, suggestion = model.kneighbors(
    book_pivot.iloc[43, :].values.reshape(1, -1), n_neighbors=6)

book_names = book_pivot.index

ids = np.where(final_rating['Title'] == "The Lost World")[0][0]

final_rating.iloc[ids]['Img-URL']

book_name = []
for book_id in suggestion:
    book_name.append(book_pivot.index[book_id])

ids_index = []
for name in book_name[0]:
    ids = np.where(final_rating['Title'] == name)[0][0]
    ids_index.append(ids)

for idx in ids_index:
    url = final_rating.iloc[idx]['Img-URL']
    # print(url)

pickle.dump(model, open('artifacts/model.pkl', 'wb'))
pickle.dump(book_names, open('artifacts/book_names.pkl', 'wb'))
pickle.dump(final_rating, open('artifacts/final_rating.pkl', 'wb'))
pickle.dump(book_pivot, open('artifacts/book_pivot.pkl', 'wb'))


def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            if j == book_name:
                print(f"You searched '{book_name}'\n")
                print("The suggestion books are: \n")
            else:
                print(j)


book_name = "A Fine Balance"
recommend_book(book_name)
