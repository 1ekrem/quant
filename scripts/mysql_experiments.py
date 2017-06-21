'''
Created on 20 Jun 2017

@author: wayne
'''
import MySQLdb as mdb

HOST = 'localhost'
USER = 'wayne'
PASSWORD = ''


def test_list_users():
    db = mdb.connect(host=HOST,    # your host, usually localhost
                     user=USER,         # your username
                     passwd=PASSWORD,  # your password
                     db="mysql")        # name of the data base
    
    cur = db.cursor()
    
    cur.execute("SELECT User FROM mysql.user")
    
    for row in cur.fetchall():
        print row[0]
    
    db.close()


def test_creating_table():
    
    con = mdb.connect(HOST, USER, PASSWORD, 'testdb');

    with con:
        
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS Writers")
        cur.execute("CREATE TABLE Writers(Id INT PRIMARY KEY AUTO_INCREMENT, \
                     Name VARCHAR(25))")
        cur.execute("INSERT INTO Writers(Name) VALUES('Jack London')")
        cur.execute("INSERT INTO Writers(Name) VALUES('Honore de Balzac')")
        cur.execute("INSERT INTO Writers(Name) VALUES('Lion Feuchtwanger')")
        cur.execute("INSERT INTO Writers(Name) VALUES('Emile Zola')")
        cur.execute("INSERT INTO Writers(Name) VALUES('Truman Capote')")


def test_reading_dictionary_cursor():
    con = mdb.connect(HOST, USER, PASSWORD, 'testdb')
    
    with con:
    
        cur = con.cursor(mdb.cursors.DictCursor)
        cur.execute("SELECT * FROM Writers")
    
        rows = cur.fetchall()
        
        print(rows)
    
        for row in rows:
            print row["Id"], row["Name"]
