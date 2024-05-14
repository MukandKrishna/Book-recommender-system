# Book Recommendation System

## Overview
This project implements a book recommendation system using various techniques such as content filtering, collaborative filtering, Graph Neural Networks, and neural collaborative filtering (NCF). 
The goal of the recommendation system is to provide personalized book recommendations to users based on their preferences and interactions with books.

## Features
- **Content Filtering**: Recommends books similar to those the user has liked based on the content or features of the books.
- **Collaborative Filtering**: Recommends books by identifying patterns among users and items, leveraging user-item interaction data.
- **Neural Collaborative Filtering (NCF)**: Utilizes neural networks to learn user-item interactions directly from data, capturing complex patterns and relationships.
- **Graph Neural Network**: It runs on collaborative filtering data, leveraging user-item interactions to make recommendations. These interactions can be explicit, such as ratings or reviews, etc.
It then is represented as a bipartite graph, where nodes represent users and items, and edges represent interactions between them. GNNs learn to propagate information through this graph structure to capture user-item relationships and make personalized recommendations.

### Image Representation

Here is an image showing the books and users relationship:

![Books-users](https://github.com/MukandKrishna/Book-recommender-system/raw/main/images/Books-users.png)

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TensorFlow/Keras (for NCF)
- GNN ( Collaborative Filtering )

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
3. Run 4 separated codes in the ***'Main Code (Content, Collaborative, GNN and NCF)'*** folder:

    ```
    content-filtering.ipynb
    Collaborative Filtering.ipynb
    NCF.ipynb
    GNN.ipynb
    ```

## Usage
1. **Content Filtering**: Users can receive recommendations based on books similar to those they have liked in the past. 
2. **Collaborative Filtering**: Users can receive recommendations based on patterns among users and items, leveraging user-item interaction data.
3. **Graph Neural Network**: It runs on collaborative filtering data, leveraging user-item interactions to make recommendations. These interactions can be explicit, such as ratings or reviews, etc.
It then is represented as a bipartite graph, where nodes represent users and items, and edges represent interactions between them. GNNs learn to propagate information through this graph structure to capture user-item relationships and make personalized recommendations.
5. **NCF**: Users can receive recommendations using neural collaborative filtering, capturing complex patterns and relationships directly from data.

## Examples
- **Content Filtering**: Recommend books similar to "The Catcher in the Rye".
- **Collaborative Filtering**: Recommend books for user ID 1234.
- **NCF**: Recommend using neural collaborative filtering books for user ID 5678.
- **GNN**: Recommend ISBNs using neural collaborative filtering books for user ID 10.

## Contributors
- [Mukand Krishna](https://github.com/MukandKrishna) [Faris Asif](https://github.com/farisasif7) [Insia Farhan](https://github.com/K200265-Insia-Farhan)
