import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
import hashlib
from thefuzz import fuzz, process

# PostgreSQL connection string (replace with your actual credentials)
DATABASE_URL = "postgresql://postgres:admin@localhost/book_search"

# Create a connection to the database
engine = create_engine(DATABASE_URL)
conn = engine.connect()

# Function to hash passwords using MD5
def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

# Function to validate user login
def validate_login(username, password):
    hashed_password = hash_password(password)
    
    # Query to check if the username and hashed password exist in the credentials table
    query = text("SELECT * FROM credentials WHERE username = :username AND pswd = :password")
    result = conn.execute(query, {"username": username, "password": hashed_password}).fetchone()

    return result is not None

# Check if user exists
def check_user_exists(username):
    query = text("SELECT * FROM credentials WHERE username = :username")
    result = conn.execute(query, {"username": username}).fetchone()
    return result is not None

# Function to add a user
def add_user(username, password):
    hashed_password = hash_password(password)
    query = text("INSERT INTO credentials (username, pswd) VALUES (:username, :password)")
    conn.execute(query, {"username": username, "password": hashed_password})
    conn.commit()  # Ensure the changes are committed to the database

# Function to update a user's password
def update_password(username, new_password):
    hashed_password = hash_password(new_password)
    query = text("UPDATE credentials SET pswd = :password WHERE username = :username")
    conn.execute(query, {"password": hashed_password, "username": username})
    conn.commit()  # Ensure the changes are committed to the database

# Fetch books from the booksinfo table
def fetch_books():
    result_set = conn.execute(text("SELECT * FROM booksinfo"))
    df = pd.DataFrame(result_set.fetchall(), columns=result_set.keys())
    return df

# Function to add a book to the booksinfo table
def add_book(book_title, book_price, rating, author, year_of_publication, genre, url, cover_image_url):
    query = text("""
        INSERT INTO booksinfo (book_title, book_price, rating, author, year_of_publication, genre, url, cover_image_url)
        VALUES (:book_title, :book_price, :rating, :author, :year_of_publication, :genre, :url, :cover_image_url)
    """)
    conn.execute(query, {
        "book_title": book_title,
        "book_price": book_price,
        "rating": rating,
        "author": author,
        "year_of_publication": year_of_publication,
        "genre": genre,
        "url": url,
        "cover_image_url": cover_image_url
    })
    conn.commit()  # Ensure the changes are committed to the database

# Normalizing book features for KNN
def normalize_features(df):
    df['year_of_publication'] = pd.to_numeric(df['year_of_publication'], errors='coerce').fillna(0)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    
    # Scaling the features
    scaler = StandardScaler()
    feature_columns = ['rating', 'year_of_publication']
    
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df, feature_columns

# KNN recommendation system based on selected book title and genre
def knn_recommend_books(selected_book_title, selected_book_genre, n_neighbors=5):
    books_df = fetch_books()
    books_df, feature_columns = normalize_features(books_df)

    # Create KNN model
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(books_df[feature_columns])

    selected_book = books_df[books_df['book_title'].str.contains(selected_book_title, case=False) & 
                             (books_df['genre'].str.contains(selected_book_genre, case=False))]
    
    if selected_book.empty:
        return pd.DataFrame()  # Return an empty dataframe if no book is found
    
    selected_book_index = selected_book.index[0]

    # Get the nearest neighbors
    distances, indices = knn.kneighbors([books_df.loc[selected_book_index, feature_columns]])

    # Retrieve recommended books and filter out those already shown
    recommended_books = books_df.iloc[indices[0]].reset_index(drop=True)
    return recommended_books[~recommended_books['book_title'].isin([selected_book.iloc[0]['book_title']])]

# Fuzzy search for books based on title, author, genre, or year
from fuzzywuzzy import process, fuzz
import pandas as pd

def fuzzy_search_books(search_term):
    books_df = fetch_books()
    search_term = search_term.lower()  # Convert to lowercase for case-insensitive matching
    threshold = 70  # Define similarity threshold

    # Create a dictionary to collect matched rows with their scores
    matches = {'book_title': [], 'author': [], 'genre': [], 'year_of_publication': []}

    # Check fuzzy matches for each column
    for column in matches.keys():
        if column == 'year_of_publication':
            # Convert year to string before matching
            column_matches = process.extract(search_term, books_df[column].astype(str), scorer=fuzz.token_set_ratio)
        else:
            column_matches = process.extract(search_term, books_df[column], scorer=fuzz.token_set_ratio)

        # Filter matches based on the threshold and add unique matches to the list
        matches[column] = [match[0] for match in column_matches if match[1] >= threshold]

    # Combine all unique matched book IDs
    matched_books = books_df[
        books_df['book_title'].isin(matches['book_title']) |
        books_df['author'].isin(matches['author']) |
        books_df['genre'].isin(matches['genre']) |
        books_df['year_of_publication'].astype(str).isin(matches['year_of_publication'])
    ]

    # If matched_books DataFrame is empty, try a more lenient threshold
    if matched_books.empty:
        lower_threshold = 60
        for column in matches.keys():
            if column == 'year_of_publication':
                column_matches = process.extract(search_term, books_df[column].astype(str), scorer=fuzz.partial_ratio)
            else:
                column_matches = process.extract(search_term, books_df[column], scorer=fuzz.partial_ratio)
            matches[column] = [match[0] for match in column_matches if match[1] >= lower_threshold]

        # Combine all unique matched book IDs with the lower threshold
        matched_books = books_df[
            books_df['book_title'].isin(matches['book_title']) |
            books_df['author'].isin(matches['author']) |
            books_df['genre'].isin(matches['genre']) |
            books_df['year_of_publication'].astype(str).isin(matches['year_of_publication'])
        ]

    return matched_books.head(10) if not matched_books.empty else pd.DataFrame()

# Highlight matched text
def highlight_matched_text(text, search_term):
    if search_term:
        return text.replace(search_term, f"<span style='background-color: yellow;'>{search_term}</span>")
    return text

from sklearn.metrics.pairwise import cosine_similarity

# Cosine Similarity recommendation system based on selected book title and genre
def cosine_recommend_books(selected_book_title, selected_book_genre, n_neighbors=5):
    books_df = fetch_books()
    books_df, feature_columns = normalize_features(books_df)

    # Create cosine similarity model
    cos_sim_matrix = cosine_similarity(books_df[feature_columns])
    
    selected_book = books_df[books_df['book_title'].str.contains(selected_book_title, case=False) & 
                             (books_df['genre'].str.contains(selected_book_genre, case=False))]
    
    if selected_book.empty:
        return pd.DataFrame()  # Return an empty dataframe if no book is found
    
    selected_book_index = selected_book.index[0]

    # Get the nearest neighbors
    similarity_scores = list(enumerate(cos_sim_matrix[selected_book_index]))
    
    # Sort books by similarity score
    sorted_similar_books = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:n_neighbors + 1]

    # Retrieve recommended books
    recommended_indices = [i[0] for i in sorted_similar_books]
    recommended_books = books_df.iloc[recommended_indices].reset_index(drop=True)

    return recommended_books

# Hybrid recommendation system combining KNN and Cosine Similarity
def hybrid_recommend_books(selected_book_title, selected_book_genre, n_neighbors=5):
    knn_recommendations = knn_recommend_books(selected_book_title, selected_book_genre, n_neighbors)
    cosine_recommendations = cosine_recommend_books(selected_book_title, selected_book_genre, n_neighbors)
    
    # Combine the results
    combined_recommendations = pd.concat([knn_recommendations, cosine_recommendations]).drop_duplicates().reset_index(drop=True)
    
    return combined_recommendations

# Book search page with fuzzy search and hybrid recommendations
def book_search_page():
    st.title(f"Welcome, {st.session_state['username']}!")

    search_term = st.text_input("Search for a book by title, author, genre, or year:")

    if st.button("Search"):
        if search_term:
            filtered_df = fuzzy_search_books(search_term)

            if not filtered_df.empty:
                num_columns = 3
                total_books = len(filtered_df)

                for idx in range(0, total_books, num_columns):
                    cols = st.columns(num_columns)
                    for col_index in range(num_columns):
                        if idx + col_index < total_books:
                            row = filtered_df.iloc[idx + col_index]

                            with cols[col_index]:
                                if row['cover_image_url']:
                                    st.image(row['cover_image_url'], width=170)
                                
                                st.write(f"*Book Title:* {highlight_matched_text(row['book_title'], search_term)}", unsafe_allow_html=True)
                                st.write(f"*Author:* {highlight_matched_text(row['author'], search_term)}", unsafe_allow_html=True)
                                st.write(f"*Year of Publication:* {highlight_matched_text(str(row['year_of_publication']), search_term)}", unsafe_allow_html=True)
                                st.write(f"*Genre:* {highlight_matched_text(row['genre'], search_term)}", unsafe_allow_html=True)
                                st.write(f"*Book Price:* {highlight_matched_text(str(row['book_price']), search_term)}", unsafe_allow_html=True)
                                st.write(f"*Rating:* {highlight_matched_text(str(row['rating']), search_term)}", unsafe_allow_html=True)
                                formatted_url = row['url']
                                if not formatted_url.startswith("http://") and not formatted_url.startswith("https://"):
                                    formatted_url = "http://" + formatted_url
                                st.write(f"URL: [Link to Book]({formatted_url})", unsafe_allow_html=True)

                    st.write("---")
                
                 # Recommend books using hybrid approach based on first result's title and genre
                top_book_title = filtered_df.iloc[0]['book_title']
                top_book_genre = filtered_df.iloc[0]['genre']
                st.title("You might also like:")
                recommended_books = hybrid_recommend_books(top_book_title, top_book_genre)

                if not recommended_books.empty:
                    total_recommendations = len(recommended_books)

                    for idx in range(0, total_recommendations, num_columns):
                        cols = st.columns(num_columns)
                        for col_index in range(num_columns):
                            if idx + col_index < total_recommendations:
                                row = recommended_books.iloc[idx + col_index]

                                with cols[col_index]:
                                    if row['cover_image_url']:
                                        st.image(row['cover_image_url'], width=100)
                                    st.write(f"*Recommended Book:* {highlight_matched_text(row['book_title'], search_term)} ({row['genre']})", unsafe_allow_html=True)
                                    recommended_url = row['url']
                                    if not recommended_url.startswith("http://") and not recommended_url.startswith("https://"):
                                        recommended_url = "http://" + recommended_url
                                    st.write(f"URL: [Link to Book]({recommended_url})", unsafe_allow_html=True)

                        st.write("---")
                else:
                    st.write("No additional recommendations available.")
            else:
                st.warning("No books found with the given search term.")
        else:
            st.warning("Please enter a search term.")


    # Button to navigate to "Add a New Book" page
    if st.button("Add a New Book"):
        st.session_state['page'] = 'add_book'

    # Logout button
    if st.button("Logout"):
        logout()

# Add a new book page
def add_book_page():
    st.title("Add a New Book")
    
    book_title = st.text_input("Book Title")
    book_price = st.number_input("Book Price", min_value=0.0)
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
    author = st.text_input("Author")
    year_of_publication = st.text_input("Year of Publication")
    genre = st.text_input("Genre")
    url = st.text_input("Book URL")
    cover_image_url = st.text_input("Cover Image URL (optional)")

    if st.button("Add Book"):
        if book_title and author and year_of_publication and genre:
            add_book(book_title, book_price, rating, author, year_of_publication, genre, url, cover_image_url)
            st.success(f"Book '{book_title}' added successfully!")
        else:
            st.warning("Please fill in all the required fields.")

    # Button to go back to the book search page
    if st.button("Back to Search"):
        st.session_state['page'] = 'search'

    # Logout button
    if st.button("Logout"):
        logout()

# Logout function
def logout():
    st.session_state.clear()  # Clear the entire session state
    st.session_state['page'] = 'auth'  # Redirect to the authentication page
    st.success("Logged out successfully!")

# Authentication page (Login/Signup/Forgot Password)
def auth_page():
    st.title("Welcome to IntelliBook!")
    st.markdown("""This application helps you find books intelligently by allowing you to search by various details such as title, author, genre, and more. Enjoy your reading journey!""")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if validate_login(username, password):
            st.session_state['username'] = username
            st.session_state['page'] = 'search'
            st.success(f"Welcome back, {username}!")
        else:
            st.error("Invalid username or password. Please try again.")

    if st.button("Sign Up"):
        if username and password:  # Ensure both fields are filled
            if not check_user_exists(username):
                add_user(username, password)
                st.success("Sign up successful! You can now log in.")
            else:
                st.error("Username already exists. Please choose a different one.")
        else:
            st.error("Please enter both username and password.")  # Error message for empty fields

    if st.button("Forgot Password"):
        new_password = st.text_input("Enter new password", type="password")
        if new_password:
            update_password(username, new_password)
            st.success("Password updated successfully!")


# Function to handle page routing
def main():
    # Page routing
    if 'page' not in st.session_state:
        st.session_state['page'] = 'auth'
    
    if st.session_state['page'] == 'auth':
        auth_page()
    elif st.session_state['page'] == 'search':
        book_search_page()
    elif st.session_state['page'] == 'add_book':
        add_book_page()

if __name__ == "__main__":
    main()