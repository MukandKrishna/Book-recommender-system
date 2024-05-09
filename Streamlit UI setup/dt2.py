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
csv_file = 'data/Ratings.csv'

# Open the CSV file and read its contents
with open(csv_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    # Iterate over each row in the CSV file
    for row in reader:
        isbn = row['ISBN']

        # Check if the ISBN value exists in the books table
        cursor.execute("SELECT COUNT(*) FROM books WHERE ISBN = %s", (isbn,))
        count = cursor.fetchone()[0]

        if count == 0:
            print(
                f"Skipping insertion for ISBN {isbn}: Not found in the books table.")
            continue

        # Prepare the INSERT statement
        insert_query = """
            INSERT INTO public.ratings
            ("User-ID", ISBN, "Book-Rating")
            VALUES (%s, %s, %s);
        """

        # Execute the INSERT statement
        cursor.execute(insert_query, (
            row['User-ID'],
            row['ISBN'],  # Using row['ISBN'] directly
            row['Book-Rating']
        ))

# Commit changes and close connection
conn.commit()
conn.close()
