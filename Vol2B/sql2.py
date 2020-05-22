import sqlite3 as sql
import csv
import pandas as pd

def prob1():
    """
        Specify relationships between columns in given sql tables.
        """
    print "One-to-one relationships:"
    # Put print statements specifying one-to-one relationships between table
    # columns.
    print "Student ID to Name (Table 5.1);"
    print "Student ID to MajorCode (Table 5.1);"
    print "Student ID to MinorCode (Table 5.1);"
    print "Name to MajorCode (Table 5.1);"
    print "Name to MinorCode (Table 5.1);"
    print "ID to Name (Table 5.2);"
    print "ClassID to Name (Table 5.4);"
    
    print "**************************"
    print "One-to-many relationships:"
    # Put print statements specifying one-to-many relationships between table
    # columns.
    
    
    print "***************************"
    print "Many-to-Many relationships:"
    # Put print statements specifying many-to-many relationships between table
    # columns.
    print "MajorCode to MinorCode (Table 5.1);"
    print "Student ID to ClassID (Table 5.3);"
    print "Student ID to Grade (Table 5.3);"

def prob2():
    """
        Write a SQL query that will output how many students belong to each major,
        including students who don't have a major.
        
        Return: A table indicating how many students belong to each major.
        """
    #Build your tables and/or query here
    query = 'SELECT FieldName, COUNT(StudentId) FROM students LEFT OUTER JOIN fields ON students.MajorCode=fields.FieldID GROUP BY FieldName ORDER BY FieldName ASC;'
    with open('students.csv') as cfile :
        rows1 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('fields.csv') as cfile :
        rows2 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('grades.csv') as cfile :
        rows3 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('classes.csv') as cfile :
        rows4 = [row for row in csv.reader(cfile,delimiter=',')]
    db = sql.connect('sql2')
    cur = db.cursor()
    cur.execute('DROP TABLE IF EXISTS students;')
    cur.execute('CREATE TABLE students (StudentId INTEGER NOT NULL,Name TEXT,MajorCode TEXT,MinorCode TEXT);')
    cur.executemany('INSERT INTO students VALUES(?,?,?,?);',rows1)
    cur.execute('DROP TABLE IF EXISTS fields;')
    cur.execute('CREATE TABLE fields (FieldID TEXT,FieldName TEXT);')
    cur.executemany('INSERT INTO fields VALUES(?,?);',rows2)
    cur.execute('DROP TABLE IF EXISTS grades;')
    cur.execute('CREATE TABLE grades (StudentId TEXT,ClassId TEXT,Grade TEXT);')
    cur.executemany('INSERT INTO grades VALUES(?,?,?);',rows3)
    cur.execute('DROP TABLE IF EXISTS classes;')
    cur.execute('CREATE TABLE classes (ClassID TEXT,ClassName TEXT);')
    cur.executemany('INSERT INTO classes VALUES(?,?);',rows4)
    db.commit()
    # This line will make a pretty table with the results of your query.
    ### query is a string containing your sql query
    ### db is a sql database connection
    result =  pd.read_sql_query(query, db)
    db.close()
    return result


def prob3():
    """
        Select students who received two or more non-Null grades in their classes.
        
        Return: A table of the students' names and the grades each received.
        """
    #Build your tables and/or query here
    query = 'SELECT Name, COUNT(GRADE) FROM students JOIN grades ON students.StudentId=grades.StudentId WHERE Grade IS NOT "NULL" GROUP BY Name HAVING COUNT(GRADE)>2;'
    with open('students.csv') as cfile :
        rows1 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('fields.csv') as cfile :
        rows2 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('grades.csv') as cfile :
        rows3 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('classes.csv') as cfile :
        rows4 = [row for row in csv.reader(cfile,delimiter=',')]
    db = sql.connect('sql2')
    cur = db.cursor()
    cur.execute('DROP TABLE IF EXISTS students;')
    cur.execute('CREATE TABLE students (StudentId INTEGER NOT NULL,Name TEXT,MajorCode TEXT,MinorCode TEXT);')
    cur.executemany('INSERT INTO students VALUES(?,?,?,?);',rows1)
    cur.execute('DROP TABLE IF EXISTS fields;')
    cur.execute('CREATE TABLE fields (FieldID TEXT,FieldName TEXT);')
    cur.executemany('INSERT INTO fields VALUES(?,?);',rows2)
    cur.execute('DROP TABLE IF EXISTS grades;')
    cur.execute('CREATE TABLE grades (StudentId TEXT,ClassId TEXT,Grade TEXT);')
    cur.executemany('INSERT INTO grades VALUES(?,?,?);',rows3)
    cur.execute('DROP TABLE IF EXISTS classes;')
    cur.execute('CREATE TABLE classes (ClassID TEXT,ClassName TEXT);')
    cur.executemany('INSERT INTO classes VALUES(?,?);',rows4)
    db.commit()
    # This line will make a pretty table with the results of your query.
    ### query is a string containing your sql query
    ### db is a sql database connection
    result =  pd.read_sql_query(query, db)
    db.close()
    return result


def prob4():
    """
        Get the average GPA at the school using the given tables.
        
        Return: A float representing the average GPA, rounded to 2 decimal places.
        """
    #2.81
    #pass
    with open('students.csv') as cfile :
        rows1 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('fields.csv') as cfile :
        rows2 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('grades.csv') as cfile :
        rows3 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('classes.csv') as cfile :
        rows4 = [row for row in csv.reader(cfile,delimiter=',')]
    db = sql.connect('sql2')
    cur = db.cursor()
    cur.execute('DROP TABLE IF EXISTS students;')
    cur.execute('CREATE TABLE students (StudentId INTEGER NOT NULL,Name TEXT,MajorCode TEXT,MinorCode TEXT);')
    cur.executemany('INSERT INTO students VALUES(?,?,?,?);',rows1)
    cur.execute('DROP TABLE IF EXISTS fields;')
    cur.execute('CREATE TABLE fields (FieldID TEXT,FieldName TEXT);')
    cur.executemany('INSERT INTO fields VALUES(?,?);',rows2)
    cur.execute('DROP TABLE IF EXISTS grades;')
    cur.execute('CREATE TABLE grades (StudentId TEXT,ClassId TEXT,Grade TEXT);')
    cur.executemany('INSERT INTO grades VALUES(?,?,?);',rows3)
    cur.execute('DROP TABLE IF EXISTS classes;')
    cur.execute('CREATE TABLE classes (ClassID TEXT,ClassName TEXT);')
    cur.executemany('INSERT INTO classes VALUES(?,?);',rows4)
    cur.execute('SELECT AVG(CASE Grade WHEN "A+" THEN 4.0 WHEN "A" THEN 4.0 WHEN "A-" THEN 4.0 WHEN "B+" THEN 3.0 WHEN "B" THEN 3.0 WHEN "B-" THEN 3.0 WHEN "C+" THEN 2.0 WHEN "C" THEN 2.0 WHEN "C-" THEN 2.0 WHEN "D+" THEN 1.0 WHEN "D" THEN 1.0 WHEN "D-" THEN 1.0 WHEN Grade IS "NULL" THEN 0.0 END) FROM grades;')
    ans = cur.fetchone()[0]
    db.commit()
    db.close()
    return round(ans,2)


def prob5():
    """
        Find all students whose last name begins with 'C' and their majors.
        
        Return: A table containing the names of the students and their majors.
        """
    #Build your tables and/or query here
    query = 'SELECT Name, FieldName FROM students LEFT OUTER JOIN fields ON students.MajorCode=fields.FieldID WHERE Name LIKE "% C%";'
    with open('students.csv') as cfile :
        rows1 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('fields.csv') as cfile :
        rows2 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('grades.csv') as cfile :
        rows3 = [row for row in csv.reader(cfile,delimiter=',')]
    with open('classes.csv') as cfile :
        rows4 = [row for row in csv.reader(cfile,delimiter=',')]
    db = sql.connect('sql2')
    cur = db.cursor()
    cur.execute('DROP TABLE IF EXISTS students;')
    cur.execute('CREATE TABLE students (StudentId INTEGER NOT NULL,Name TEXT,MajorCode TEXT,MinorCode TEXT);')
    cur.executemany('INSERT INTO students VALUES(?,?,?,?);',rows1)
    cur.execute('DROP TABLE IF EXISTS fields;')
    cur.execute('CREATE TABLE fields (FieldID TEXT,FieldName TEXT);')
    cur.executemany('INSERT INTO fields VALUES(?,?);',rows2)
    cur.execute('DROP TABLE IF EXISTS grades;')
    cur.execute('CREATE TABLE grades (StudentId TEXT,ClassId TEXT,Grade TEXT);')
    cur.executemany('INSERT INTO grades VALUES(?,?,?);',rows3)
    cur.execute('DROP TABLE IF EXISTS classes;')
    cur.execute('CREATE TABLE classes (ClassID TEXT,ClassName TEXT);')
    cur.executemany('INSERT INTO classes VALUES(?,?);',rows4)
    db.commit()
    # This line will make a pretty table with the results of your query.
    ### query is a string containing your sql query
    ### db is a sql database connection
    result =  pd.read_sql_query(query, db)
    db.close()
    return result
