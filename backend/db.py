import sqlite3
from datetime import datetime

conn = sqlite3.connect('building_counts.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS building_count (
    building_id TEXT PRIMARY KEY,
    building_name TEXT NOT NULL,
    current_count INTEGER,
    last_updated TEXT
)
''')
conn.commit()

# Insert your data if not present
initial_data = [
    ('b_1', 'Coridor', 100),
    ('b_2', 'Department of Chemical and Process Engineering', 200),
    ('b_3', 'Department Engineering Mathematics/Department Engineering Management/Computer Center', 150),
    ('b_4', 'Drawing Office 1', 80),
    ('b_5', 'Professor E.O.E. Pereira Theatre', 300),
    ('b_6', 'Administrative Building', 100),
    ('b_7', 'Security Unit', 50),
    ('b_8', 'Electronic Lab', 60),
    ('b_9', 'Department of Electrical and Electronic Engineering', 250),
    ('b_10', 'Department of Computer Engineering', 80),
    ('b_11', 'Electrical and Electronic Workshop', 40),
    ('b_12', 'Surveying Lab', 70),
    ('b_13', 'Soil Lab', 90),
    ('b_14', 'Materials Lab', 120),
    ('b_15', 'Environmental Lab', 40),
    ('b_16', 'Fluids Lab', 100),
    ('b_17', 'New Mechanics Lab', 60),
    ('b_18', 'Applied Mechanics Lab', 50),
    ('b_19', 'Thermodynamics Lab', 110),
    ('b_20', 'Generator Room', 20),
    ('b_21', 'Engineering Workshop', 100),
    ('b_22', 'Engineering Carpentry Shop', 40),
    ('b_23', 'Drawing Office 2', 80),
    ('b_24', 'Lecture Room (middle-right)', 60),
    ('b_25', 'Structures Laboratory', 70),
    ('b_26', 'Lecture Room (bottom-right)', 60),
    ('b_27', 'Engineering Liboratory', 100),
    ('b_28', 'Department of Manufacturing and Industrial Engineering', 120),
    ('b_29', 'Faculty Canteen', 80),
    ('b_30', 'High Voltage Laboratory', 40)
]

for building_id, building_name, count in initial_data:
    cursor.execute('''
        INSERT OR IGNORE INTO building_count (building_id, building_name, current_count, last_updated)
        VALUES (?, ?, ?, ?)
    ''', (building_id, building_name, count, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
conn.commit()

def get_all_counts():
    cursor.execute('SELECT building_id, building_name, current_count FROM building_count')
    return cursor.fetchall()
