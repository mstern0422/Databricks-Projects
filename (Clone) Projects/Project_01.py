# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 417 - Project 01
# MAGIC ## Analysis of NASA Server Logs
# MAGIC **Mallory Stern**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: Set up Environment
# MAGIC In the first portion of the project, the environment will be set up by using import statements.

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC The following will now create our SparkSession and SparkContext objects.

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B: Load and Process Data
# MAGIC For this project, a file that contains one month of server log data collected from NASA.gov in August 1995 will be utilized. The following code will load this data file into an RDD and display the number of elements contained in the created RDD.

# COMMAND ----------

nasa_raw = sc.textFile('/FileStore/tables/NASA_server_logs_Aug_1995.txt')
print(nasa_raw.count())

# COMMAND ----------

# MAGIC %md
# MAGIC The first 10 elements of this RDD will now be displayed.

# COMMAND ----------

for i in nasa_raw.take(10):
    print(i)

# COMMAND ----------

# MAGIC %md
# MAGIC Each line of the server information will now be processed by removing the double quotes, tokenizing the strings, replacing the hyphens that appear for the number of bytes with zeros, and coercing the values into appropriate datatypes. The first 10 elements from this list will then be displayed.

# COMMAND ----------

def process_row(row):
    item = row.replace('"', '').split(' ')
    if item[-1] == '-':
        item[-1] = '0'
    item[-1] = int(item[-1])
    return item

nasa = nasa_raw.map(process_row).persist()

for i in nasa.take(10):
    print(i)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C: Most Requested Resources
# MAGIC In this part of the project, the most frequently requested resources will be determined. 

# COMMAND ----------

count_by_resource = nasa.map(lambda x: (x[3], 1)).reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[1], ascending = False)

for i in count_by_resource.take(10):
    print(i)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part D: Most Common Request Origins
# MAGIC For each of the requests in this dataset, the IP address or DNS hostname for the server from which the request originated are provided. In the following portion of the project, which servers are the origins for the greatest number of requests will be determined.

# COMMAND ----------

count_by_origin = nasa.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[1], ascending = False)

for i in count_by_origin.take(10):
    print(i)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part E: Request Types
# MAGIC This portion of the project will now analyze the reconrds based on their request type. The following will confirm that there are three different request types: GET, HEAD, and POST.

# COMMAND ----------

req_types = nasa.map(lambda x: x[2]).distinct().collect()

print(req_types)

# COMMAND ----------

# MAGIC %md
# MAGIC The number of each request type will now be counted.

# COMMAND ----------

for req_type in req_types:
    if req_type == 'GET':
        print('There were', nasa.filter(lambda x: x[2] == req_type).count(), 'GET requests.')
    elif req_type == 'HEAD':
        print('There were', nasa.filter(lambda x: x[2] == req_type).count(), 'HEAD requests.')
    else:
        print('There were', nasa.filter(lambda x: x[2] == req_type).count(), 'POST requests.')

# COMMAND ----------

# MAGIC %md
# MAGIC Now the average number of bytes returned to the client for each request type will be determined. 

# COMMAND ----------

avg_bytes = nasa.map(lambda x: (x[2], (x[5], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).map(lambda x: (x[0], int(x[1][0]/x[1][1])))

for i in avg_bytes.collect():
    print(i)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part F: Status Codes
# MAGIC The status codes returned by the server will be analyzed in the following portion of the project. 

# COMMAND ----------

status_codes = nasa.map(lambda x: x[4]).distinct().sortBy(lambda x: x).collect()

print(status_codes)

# COMMAND ----------

# MAGIC %md
# MAGIC The following will determine which status codes appear for each request type.

# COMMAND ----------

for req_type in req_types:
    if req_type == 'GET':
        print('Status codes for GET requests:', nasa.filter(lambda x: x[2] == req_type).filter(lambda x: (x[4] in status_codes)).map(lambda x: x[4]).distinct().sortBy(lambda x: x).collect())
    elif req_type == 'HEAD':
        print('Status codes for HEAD requests:', nasa.filter(lambda x: x[2] == req_type).filter(lambda x: (x[4] in status_codes)).map(lambda x: x[4]).distinct().sortBy(lambda x: x).collect())
    else:
        print('Status codes for POST requests:', nasa.filter(lambda x: x[2] == req_type).filter(lambda x: (x[4] in status_codes)).map(lambda x: x[4]).distinct().sortBy(lambda x: x).collect())

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will count the number of requests resulting in each status code.

# COMMAND ----------

code_counts = []

for status_code in status_codes:
    code_counts.append(nasa.filter(lambda x: x[4] == status_code).count())

plt.figure(figsize=[10,4])
plt.bar(status_codes, code_counts, color='lavender', edgecolor='k')
plt.xlabel('Status Code')
plt.ylabel('Count')
plt.yscale('log')
plt.title('Distribution of Status Codes')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part G: Request Volume by Day
# MAGIC The final portion of the project will determine the number of requests recieved by the server during each day in August 1995.

# COMMAND ----------

counts_by_day = nasa.map(lambda x: (x[1][1:3], 1)).reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[0])

for i in counts_by_day.take(5):
    print(i)

# COMMAND ----------

# MAGIC %md
# MAGIC A bar chart will be created that will display the distribution of requests by day of the month.

# COMMAND ----------

day_list = counts_by_day.map(lambda x: x[0]).collect()
count_list = counts_by_day.map(lambda x: x[1]).collect()

plt.figure(figsize=[10,4])
plt.bar(day_list, count_list, color='lightgreen', edgecolor='k')
plt.xlabel('Day of Month')
plt.ylabel('Count')
plt.title('Number of Requests by Day (August 1995)')

plt.show()
