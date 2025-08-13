# Databricks notebook source
# MAGIC %md #DSCI 417 - Homework 01
# MAGIC **Mallory Stern**

# COMMAND ----------

from pyspark.sql import SparkSession
import math
from pyspark.mllib.random import RandomRDDs

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 1: Terminology

# COMMAND ----------

# MAGIC %md 
# MAGIC 1. Scala
# MAGIC 2. SparkSession instance
# MAGIC 3. SparkContext
# MAGIC 4. Resilient Distributed Datasets
# MAGIC 5. Partitions
# MAGIC 6. Transformation
# MAGIC 7. Transformation
# MAGIC 8. Transformation
# MAGIC 9. Action
# MAGIC 10. Transformation
# MAGIC 11. Action
# MAGIC 12. Python list
# MAGIC 13. Master node
# MAGIC 14. Workers
# MAGIC 15. Driver
# MAGIC 16. Executor

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 2: Working with a Numerical RDD

# COMMAND ----------

random_rdd = RandomRDDs.uniformRDD(sc, size = 1200000, seed = 1)

print("Sum:     ", random_rdd.sum())
print("Mean:    ", random_rdd.mean())
print("Std Dev: ", random_rdd.stdev())
print("Minimum: ", random_rdd.min())
print("Maximum: ", random_rdd.max())

# COMMAND ----------

print("Number of Partitions: ", random_rdd.getNumPartitions())
size_rdd = random_rdd.glom().map(len)

print(size_rdd.collect())


# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 3: Transformations

# COMMAND ----------

scaled_rdd = random_rdd.map(lambda x: x*10)

print("Sum:     ", scaled_rdd.sum())
print("Mean:    ", scaled_rdd.mean())
print("Std Dev: ", scaled_rdd.stdev())
print("Minimum: ", scaled_rdd.min())
print("Maximum: ", scaled_rdd.max())

# COMMAND ----------

log_rdd = scaled_rdd.map(lambda x: math.log(x))

print("Sum:     ", log_rdd.sum())
print("Mean:    ", log_rdd.mean())
print("Std Dev: ", log_rdd.stdev())
print("Minimum: ", log_rdd.min())
print("Maximum: ", log_rdd.max())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 4: Calculating SSE

# COMMAND ----------

pairs_raw = sc.textFile('/FileStore/tables/pairs_data.txt')
print(pairs_raw.count())

# COMMAND ----------

for pair in pairs_raw.take(5):
    print(pair)

# COMMAND ----------

def process_line(row):
    item = row.split(' ')

    return [float(item[0]), float(item[1])]

pairs = pairs_raw.map(process_line)

for i in pairs.take(5):
    print(i)

# COMMAND ----------

SSE = pairs.map(lambda x: (x[0] - x[1])**2).sum()

print(SSE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 5: Calculating r-Squared

# COMMAND ----------

mean = pairs.map(lambda x: x[0]).mean()

print(mean)

# COMMAND ----------

SST = pairs.map(lambda x: (x[0] - mean)**2).sum()

print(SST)

# COMMAND ----------

r2 = 1 - (SSE/SST)

print(r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 6: NASA Server Logs

# COMMAND ----------

nasa = sc.textFile('/FileStore/tables/NASA_server_logs_Aug_1995.txt')
print(nasa.count())

# COMMAND ----------

for i in nasa.take(5):
    print(i)

# COMMAND ----------

print("Number of GET requests:  ", nasa.map(lambda x: 'GET' in x).sum())
print("Number of POST requests: ", nasa.map(lambda x: 'POST' in x).sum())
print("Number of HEAD requests: ", nasa.map(lambda x: 'HEAD' in x).sum())
