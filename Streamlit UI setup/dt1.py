import psycopg2
import csv

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname="DLP",
    user="postgres",
    password="mukand",
    host="127.0.0.1",
    port="8000"  # Use the correct port number here
)

# Create a cursor object
cursor = conn.cursor()

# Path to the CSV file
csv_file = 'data/Books.csv'

# Open the CSV file and read its contents
with open(csv_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    # Iterate over each row in the CSV file
    for row in reader:
        # Prepare the INSERT statement
        insert_query = """
            INSERT INTO public.books
            (
            ISBN, 
            "Book-Title", 
            "Book-Author", 
            "Year-Of-Publication", 
            Publisher, 
            "Image-URL-S", 
            "Image-URL-M", 
            "Image-URL-L")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """

        # Execute the INSERT statement
        cursor.execute(insert_query, (
            row['ISBN'],
            row['Book-Title'],
            row['Book-Author'],
            row['Year-Of-Publication'],
            row['Publisher'],
            row['Image-URL-S'],
            row['Image-URL-M'],
            row['Image-URL-L']
        ))

# Commit changes and close connection
conn.commit()
conn.close()
