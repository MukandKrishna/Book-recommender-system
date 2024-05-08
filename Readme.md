# Book Recommendation System

## Overview
This project implements a book recommendation system using various techniques such as content filtering, collaborative filtering, and neural collaborative filtering (NCF). 
The goal of the recommendation system is to provide personalized book recommendations to users based on their preferences and interactions with books.

## Features
- **Content Filtering**: Recommends books similar to those the user has liked based on the content or features of the books.
- **Collaborative Filtering**: Recommends books by identifying patterns among users and items, leveraging user-item interaction data.
- **Neural Collaborative Filtering (NCF)**: Utilizes neural networks to learn user-item interactions directly from data, capturing complex patterns and relationships.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TensorFlow/Keras (for NCF)

## Getting Started
1. Clone the repository:
    ```
    git clone https://github.com/yourusername/book-recommendation-system.git
    cd book-recommendation-system
    ```
2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```
3. Run 3 separated codes in the ***'Codes'*** folder:

    ```
    content-filtering.ipynb
    Collaborative Filtering.ipynb
    NCF.ipynb
    ```

## Usage
1. **Content Filtering**: Users can receive recommendations based on books similar to those they have liked in the past. 
2. **Collaborative Filtering**: Users can receive recommendations based on patterns among users and items, leveraging user-item interaction data.
3. **NCF**: Users can receive recommendations using neural collaborative filtering, capturing complex patterns and relationships directly from data.

## Examples
- **Content Filtering**: Recommend books similar to "The Catcher in the Rye".
- **Collaborative Filtering**: Recommend books for user ID 1234.
- **NCF**: Recommend books for user ID 5678 using neural collaborative filtering.

## Contributors
- [Mukand Krishna](https://github.com/MukandKrishna)
- [Faris Asif](https://github.com/farisasif7)
- [Insia Farhan](https://github.com/name)

## License
This project is licensed under the [MIT License](LICENSE).
