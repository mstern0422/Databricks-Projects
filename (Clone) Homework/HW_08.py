# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 417 - Homework 08
# MAGIC **Mallory Stern**

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 1: Creating Streaming DataFrame

# COMMAND ----------

paysim = (spark.read.option('delimiter', ',').option('header', True).option('inferSchema', True).csv('/FileStore/tables/paysim/step_001.csv'))

paysim_schema = paysim.schema

paysim.show(5)

# COMMAND ----------

paysim_stream = (spark.readStream.option('header', True).option('delimiter', ',').option('maxFilesPerTrigger', 1).schema(paysim_schema).csv('/FileStore/tables/paysim/'))

print(paysim_stream.isStreaming)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 2: Apply Transformations

# COMMAND ----------

type_summary = (paysim_stream.groupBy('type').agg(expr('COUNT(*) AS n'), expr('ROUND(AVG(amount), 2) AS avg_amount'), expr('MIN(amount) AS min_amount'), expr('MAX(amount) AS max_amount')).orderBy('n', ascending = False))

# COMMAND ----------

destinations = (paysim_stream.filter(expr('type == "TRANSFER"')).groupBy('nameDest').agg(expr('COUNT(*) AS n'), expr('SUM(amount) AS total_amount'), expr('ROUND(AVG(amount), 2) AS avg_amount')).orderBy('n', ascending = False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 3: Define Output Sinks

# COMMAND ----------

type_writer = (type_summary.writeStream.format('memory').queryName('type_summary').outputMode('complete'))

destinations_writer = (destinations.writeStream.format('memory').queryName('destinations').outputMode('complete'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 4: Start and Monitor the Streams

# COMMAND ----------

type_query = type_writer.start()
dest_query = destinations_writer.start()

# COMMAND ----------

print(spark.sql('SELECT * FROM type_summary').count())
spark.sql('SELECT * FROM type_summary').show()

# COMMAND ----------

print(spark.sql('SELECT * FROM destinations').count())
spark.sql('SELECT * FROM destinations').show(16)

# COMMAND ----------

type_query.stop()
dest_query.stop()
