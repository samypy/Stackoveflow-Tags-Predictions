import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('sample.db')

# Create a cursor to execute SQL queries
cursor = conn.cursor()

# Create a table named "persons" with columns "id", "first_name", "last_name", and "Age"
cursor.execute('''
    CREATE TABLE persons (
        id INTEGER PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        Age INTEGER
    )
''')

# Commit the changes and close the database connection
conn.commit()
conn.close()
