# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 417 - Project 02
# MAGIC ## Student Grade Database
# MAGIC **Mallory Stern**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: Set up Environment
# MAGIC In this part of the project, the environment will be setup. We will begin with some import statements. 

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B: Load the Data
# MAGIC In this project we will be working a database of course records from a (fictional) university called Syntheticus University, which was founded in the fall of 2000. The university offers six undergraduate degree programs: Biology, Chemistry, Computer Science, Mathematics, and Physics. The data we will work with is assumed to have been collected immediately after the end of the Spring 2021 term.

# COMMAND ----------

accepted_schema = ('acc_term_id STRING, sid INTEGER, first_name STRING, last_name STRING, major STRING')
accepted = (spark.read.option('delimiter', ',').option('header', True).schema(accepted_schema).csv('/FileStore/tables/univ/accepted.csv'))

alumni_schema = ('sid INTEGER')
alumni = (spark.read.option('delimiter', '\t').option('header', True).schema(alumni_schema).csv('/FileStore/tables/univ/alumni.csv'))

expelled_schema = ('sid INTEGER')
expelled = (spark.read.option('delimiter', '\t').option('header', True).schema(expelled_schema).csv('/FileStore/tables/univ/expelled.csv'))

unretained_schema = ('sid INTEGER')
unretained = (spark.read.option('delimiter', '\t').option('header', True).schema(unretained_schema).csv('/FileStore/tables/univ/unretained.csv'))

faculty_schema = ('fid INTEGER, first_name STRING, last_name STRING, dept STRING')
faculty = (spark.read.option('delimiter', ',').option('header', True).schema(faculty_schema).csv('/FileStore/tables/univ/faculty.csv'))

course_schema = ('dept STRING, course STRING, prereq STRING, credits INTEGER')
course = (spark.read.option('delimiter', ',').option('header', True).schema(course_schema).csv('/FileStore/tables/univ/courses.csv'))

grades_schema = ('term_id STRING, course STRING, sid INTEGER, fid INTEGER, grade STRING')
grades = (spark.read.option('delimiter', ',').option('header', True).schema(grades_schema).csv('/FileStore/tables/univ/grades.csv'))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, the number of records in each DataFrame will be displayed. 

# COMMAND ----------

print('The number of records in accepted is', accepted.count(), '.')
print('The number of records in alumni is', alumni.count(), '.')
print('The number of records in expelled is', expelled.count(), '.')
print('The number of records in unretained is',unretained.count(), '.')
print('The number of records in facult is', faculty.count(), '.')
print('The number of records in course is', course.count(), '.')
print('The number of records in grades is', grades.count(), '.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C: Student Count by Status
# MAGIC In this portion of the project, the number of students will be counted in each of the following groups: students who have been accepted, students who actually enrolled in courses, current students, all former students, alumni, unretained students, and students who were expelled. 
# MAGIC
# MAGIC Three new DataFrames to store student info for students in various categories. The desired counts will then be generated. 

# COMMAND ----------

enrolled = accepted.join(other = grades, on ='sid', how = 'semi')

current = enrolled.join(other = alumni, on = 'sid', how = 'anti').join(other = unretained, on = 'sid', how = 'anti').join(other = expelled, on = 'sid', how = 'anti')

former = enrolled.join(other = current, on = 'sid', how = 'anti')

print('Number of accepted students: ', accepted.count())
print('Number of enrolled students:  ', enrolled.count())
print('Number of current students:   ', current.count())
print('Number of former students:    ', former.count())
print('Number of unretained students:', unretained.count())
print('Number of expelled students:   ', expelled.count())
print('Number of alumni:             ', alumni.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part D: Distribution of Students by Major
# MAGIC In this part, the number of students currently in each major will be determined, as well as the proportion of the overall number of students in each major.

# COMMAND ----------

NUM_STUDENTS = current.count()

current.groupBy('major').agg(expr('COUNT(sid) AS n_students'), expr('ROUND(COUNT(sid) / {}, 4) AS prop'.format(NUM_STUDENTS))).sort('prop', ascending = False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part E: Course Enrollments by Department
# MAGIC This part of the project will determine the number of students enrolled in courses offered by each department during the Spring 2021 term. This term is encoded as '2021A'.

# COMMAND ----------

sp21_enr = grades.filter(expr('term_id == "2021A"')).count()
 
( 
    grades
    .filter(expr('term_id == "2021A"'))
    .join(course, 'course', 'left')
    .groupBy('dept')
    .agg((expr('COUNT(*) AS n_students')))
    .withColumn('prop', expr(f'ROUND( n_students / {sp21_enr} ,4)') )
    .sort('n_students', ascending=False)
    .show()
)

# COMMAND ----------

sp21_enr = grades.filter(expr('term_id == "2021A"')).count()

(
grades.filter(expr('term_id == "2021A"'))
#.join(course, 'left')
# Missing the column on which the DataFrames join
.join(course, 'course', 'left')
.groupBy('dept')
.agg(expr('COUNT(*) AS n_students'), expr('ROUND(COUNT(*) / {}, 4) AS prop'.format(sp21_enr)))
.sort('prop', ascending = False).show()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part F: Graduation Rates by Major
# MAGIC This part will determine the graduation rates for each major. This analysis will be performed in steps. First, a DataFrame containing the number of former students in each major. Then, a DataFrame containing the number of alumni for each major. Lastly, these DataFrames will be combined to determine the graduation rate. 

# COMMAND ----------

former_by_major = former.groupBy('major').agg(expr('COUNT(sid) AS n_former')).sort('major', ascending = False)

former_by_major.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The number of alumni for each major will now be determined.

# COMMAND ----------

former_by_alumni = former.join(other = alumni, on = 'sid', how = 'semi').groupBy('major').agg(expr('COUNT(sid) AS n_alumni')).sort('major', ascending = False)

former_by_alumni.show()

# COMMAND ----------

former_alumni_major = former_by_alumni.join(other = former_by_major, on = 'major', how = 'outer').withColumn('grad_rate', expr('ROUND(n_alumni / n_former, 4)')).sort('major')

former_alumni_major.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part G: Number of Terms Required for Graduation 
# MAGIC In this part, a frequency distribution for the number of terms that alumni required for graduation will be calculated.

# COMMAND ----------

grades.join(other = alumni, on = 'sid', how = 'inner').groupBy('sid').agg(expr('COUNT(DISTINCT term_id) AS n_terms')).groupBy('n_terms').agg(expr('COUNT(n_terms) AS n_alumni')).sort('n_terms').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part H: Current Student GPA
# MAGIC In this section, the GPA of each current student at SU will be found and the results will be analyzed.

# COMMAND ----------

def convert_grade(grade):
    if grade == 'A':
        return 4
    elif grade == 'B':
        return 3
    elif grade == 'C':
        return 2
    elif grade == 'D':
        return 1
    elif grade == 'F':
        return 0
    else:
        return None

convert_grade_udf = udf(convert_grade)
spark.udf.register("convert_grade", convert_grade_udf)

# COMMAND ----------

# MAGIC %md
# MAGIC The GPA of each student currently enrolled at SU will now be calculated. 

# COMMAND ----------

current_gpa = grades.join(other = course, on='course', how='inner').withColumn("num_grade", convert_grade_udf("grade")).withColumn('gp', expr('credits * num_grade')).groupBy('sid').agg(expr('ROUND(SUM(gp)/SUM(credits), 2) AS gpa')).join(other = current, on = 'sid', how = 'inner').select('sid', 'first_name', 'last_name', 'major', 'gpa').sort('gpa')

current_gpa.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC The number of students with perfect 4.0 GPAs will now be determined.

# COMMAND ----------

current_gpa.filter('gpa == 4.0').show()

# COMMAND ----------

# MAGIC %md
# MAGIC The distribution of GPAs for current students will now be displayed in a histogram. 

# COMMAND ----------

current_gpa_pd = current_gpa.toPandas()
# please see sample code below
plt.hist(current_gpa_pd['gpa'], edgecolor = 'black', rwidth = 0.25, color = 'tomato')
plt.xlabel('GPA')
plt.ylabel('Count')
plt.title('GPA Distribution for Current Students')

# COMMAND ----------

gpa_pdf = current_gpa.select('gpa').toPandas()
 
plt.hist(gpa_pdf.gpa, bins=np.arange(0, 4.25, 0.25), color='cornflowerblue', edgecolor='k')
plt.xlabel('GPA')
plt.ylabel('Count')
plt.title('GPA Distribution for Current Students')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part I: Grade Distribution by Instructor 
# MAGIC This part will determine the proportion of A, B, C, D, and F grades given out by each faculty member at SU.

# COMMAND ----------

faculty_grade_dist = grades.groupBy('fid').agg(expr('COUNT(grade) AS N'),
    expr('SUM(CASE WHEN grade == "A" THEN 1 ELSE 0 END) AS countA'),
    expr('SUM(CASE WHEN grade == "B" THEN 1 ELSE 0 END) AS countB'),
    expr('SUM(CASE WHEN grade == "C" THEN 1 ELSE 0 END) AS countC'),
    expr('SUM(CASE WHEN grade == "D" THEN 1 ELSE 0 END) AS countD'),
    expr('SUM(CASE WHEN grade == "F" THEN 1 ELSE 0 END) AS countF')).join(other = faculty, on = 'fid', how = 'inner').select('fid', 'first_name', 'last_name', 'dept', 'N', expr('ROUND(countA/N, 2) AS propA'), expr('ROUND(countB/N, 2) AS propB'), expr('ROUND(countC/N, 2) AS propC'), expr('ROUND(countD/N, 2) AS propD'), expr('ROUND(countF/N, 2) AS propF'))

faculty_grade_dist.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC The faculty member who assign the fewest A grades will now be determined. There are a few faculty members who have only issued a small handful of grades, so only faculty member who have issued at least 100 grades will be considered. 

# COMMAND ----------

faculty_grade_dist.filter(expr('N >= 100')).sort('propA').show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC The faculty members who award A's most frequently will now be determined. 

# COMMAND ----------

faculty_grade_dist.filter(expr('N >= 100')).sort('propA', ascending = False).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part J: First Term GPA
# MAGIC In this section, the first-term GPA for each student who has enrolled in classes at SU will be calculated. 

# COMMAND ----------

first_term_gpa = grades.join(other = accepted, on = 'sid', how = 'inner').filter(expr('term_id == acc_term_id')).join(other = course, on = 'course', how = 'inner').withColumn("num_grade", convert_grade_udf("grade")).withColumn('gp', expr('credits * num_grade')).groupBy('sid').agg(expr('ROUND(SUM(gp)/SUM(credits), 2) AS first_term_gpa'))

first_term_gpa.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part K: Graduation Rates and First Term GPA
# MAGIC In this section, graduation rates for students whose first term GPA fals into each of four different grade ranges will be calculated. 

# COMMAND ----------

def get_gpa_range(gpa):
    if gpa >= 0 and gpa < 1:
        return '[0,1)'
    elif gpa >= 1 and gpa < 2:
        return '[1,2)'
    elif gpa >= 2 and gpa < 3:
        return '[2,3)'
    elif gpa >= 3 and gpa <= 4:
        return '[3,4]'
    else:
        return None
    
get_gpa_range_udf = udf(get_gpa_range)
spark.udf.register("get_gpa_range", get_gpa_range_udf)

# COMMAND ----------

# MAGIC %md
# MAGIC The number of alumni whose first-term GPA falls into each bin will now be calculated. 

# COMMAND ----------

alumni_ft_gpa = first_term_gpa.join(other = alumni, on = 'sid', how = 'semi').withColumn('gpa_bin', get_gpa_range_udf('first_term_gpa')).groupBy('gpa_bin').agg(expr('COUNT(sid) AS n_alumni')).sort('gpa_bin')

alumni_ft_gpa.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Next, the number of former students who's first term GPA falls into each bin will be calculated. 

# COMMAND ----------

former_ft_gpa = first_term_gpa.join(other = former, on = 'sid', how = 'semi').withColumn('gpa_bin', get_gpa_range_udf('first_term_gpa')).groupBy('gpa_bin').agg(expr('COUNT(sid) AS n_former')).sort('gpa_bin')

former_ft_gpa.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The previous two DataFrames will now be used to determin the graduation rates for each of the GPA bins.

# COMMAND ----------

alumni_ft_gpa.join(other = former_ft_gpa, on = 'gpa_bin', how = 'inner').withColumn('grad_rate', expr('ROUND(n_alumni / n_former, 4)')).sort('gpa_bin').show()

# COMMAND ----------


