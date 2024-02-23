import sqlite3
import streamlit as st

conn = sqlite3.connect('my_database.db')
# Create a cursor object
c = conn.cursor()
c.execute('SELECT * FROM requirements')

results = c.fetchall()
options = [result[1] for result in results]

selected_reqs = st.multiselect("Select the requirements to expand", options, [])

