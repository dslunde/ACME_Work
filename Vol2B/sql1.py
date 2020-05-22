import sqlite3 as sql
import csv
import pandas as pd



def prob1():
    """
    Create the following SQL tables with the following columns:
        -- MajorInfo: MajorID (int), MajorName (string)
        -- CourseInfo CourseID (int), CourseName (string)
    --------------------------------------------------------------
    Do not return anything.  Just create the designated tables.
    """
#pass
    db = sql.connect('sql1')
    cur = db.cursor()
    cur.execute('DROP TABLE IF EXISTS MajorInfo')
    cur.execute('CREATE TABLE MajorInfo (MajorID INTEGER NOT NULL, MajorName TEXT);')
    """
    cur.execute("PRAGMA table_info('MajorInfo')")
    for info in cur :
        print info
    """
    cur.execute('DROP TABLE IF EXISTS CourseInfo')
    cur.execute('CREATE TABLE CourseInfo (CourseID INTEGER NOT NULL, CourseName TEXT);')
    """
    cur.execute("PRAGMA table_info('CourseInfo')")
    for info in cur :
        print info
    """
    db.commit()
    db.close()


def prob2():
    """
    Create the following SQL table with the following columns:
        -- ICD: ID_Number (int), Gender (string), Age (int) ICD_Code (string)
    --------------------------------------------------------------
    Do not return anything.  Just create the designated table.
    """
#pass
    with open('icd9.csv','rb') as cfile :
        rows = [row for row in csv.reader(cfile,delimiter=',')]
    db = sql.connect('sql2')
    cur = db.cursor()
    cur.execute('Create Table ICD (ID_Number INTEGER NOT NULL, Gender TEXT, Age INTEGER, ICD_Code TEXT)')
    cur.executemany("INSERT INTO ICD VALUES(?,?,?,?);", rows)
#useful_test_function(db,"SELECT * FROM ICD")
    db.commit()
    db.close()

def prob3():
    """
    Create the following SQL tables with the following columns:
        -- StudentInformation: StudentID (int), Name (string), MajorCode (int)
        -- StudentGrades: StudentID (int), ClassID (int), Grade (int)

    Populate these tables, as well as the tables from Problem 1, with
        the necesary information.  Also, use the column names for
        MajorInfo and CourseInfo given in Problem 1, NOT the column
        names given in Problem 3.
    ------------------------------------------------------------------------
    Do not return anything.  Just create the designated tables.
    """
#pass
    with open('student_info.csv','rb') as cfile :
        rows1 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('major_info.csv','rb') as cfile :
        rows2 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('student_grades.csv','rb') as cfile :
        rows3 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('course_info.csv','rb') as cfile :
        rows4 = [row for row in csv.reader(cfile,delimiter=',')]
    db = sql.connect('sql1')
    cur = db.cursor()
    cur.execute('DROP TABLE IF EXISTS StudentInformation;')
    cur.execute('Create Table StudentInformation (StudentID INTEGER NOT NULL, Name TEXT, MajorCode INTEGER);')
    cur.executemany("INSERT INTO StudentInformation VALUES(?,?,?);", rows1)
    cur.execute('DROP TABLE IF EXISTS MajorInfo;')
    cur.execute('Create Table MajorInfo (ID INTEGER, Name TEXT);')
    cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", rows2)
    cur.execute('DROP TABLE IF EXISTS StudentGrades;')
    cur.execute('Create Table StudentGrades (StudentID INTEGER NOT NULL, ClassID INTEGER, Grade TEXT);')
    cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", rows3)
    cur.execute('DROP TABLE IF EXISTS CourseInfo;')
    cur.execute('Create Table CourseInfo (ClassID INTEGER, Name TEXT);')
    cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", rows4)
    useful_test_function(db,"SELECT * FROM StudentInformation")
    useful_test_function(db,"SELECT * FROM MajorInfo")
    useful_test_function(db,"SELECT * FROM StudentGrades")
    useful_test_function(db,"SELECT * FROM CourseInfo")
    db.commit()
    db.close()

def prob4():
    """
    Find the number of men and women, respectively, between ages 25 and 35
    (inclusive).
    You may assume that your "sql1" and "sql2" databases have already been
    created.
    ------------------------------------------------------------------------
    Returns:
        (n_men, n_women): A tuple containing number of men and number of women
                            (in that order)
    """
#pass
    n_men = 0
    n_women = 0
    db = sql.connect('sql2')
    cur = db.cursor()
    cur.execute("SELECT Gender FROM ICD WHERE Age>=25 AND Age<=35")
    for gen in cur :
        if gen[0] == 'M' :
            n_men += 1
        else :
            n_women += 1
    db.close()
    return n_men,n_women

def useful_test_function(db, query):
    """
    Print out the results of a query in a nice format using pandas
    ------------------------------------------------------------------------
    Inputs:
        db: A sqlite3 database connection
        query: A string containing the SQL query you want to execute
    """
    print pd.read_sql_query(query, db)
