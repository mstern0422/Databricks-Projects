# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 417 - Homework 03
# MAGIC **Mallory Stern**

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 1: Terminology

# COMMAND ----------

# MAGIC %md
# MAGIC 1. StructType
# MAGIC 2. StructField
# MAGIC 3. DoubleType
# MAGIC 4. show()
# MAGIC 5. describe() and summary()
# MAGIC 6. dropna()
# MAGIC 7. select() and withColumn()
# MAGIC 8. agg()
# MAGIC 9. col() and expr()
# MAGIC 10. filter()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 2: Columns and Expressions

# COMMAND ----------

# MAGIC %md
# MAGIC 1, 2, 5, 6, 7, 9, 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 3: Diamonds Data (Part 1)

# COMMAND ----------

diamonds_schema = ('carat DOUBLE, cut STRING, color STRING, clarity STRING, depth DOUBLE, table DOUBLE, price INTEGER, x DOUBLE, y DOUBLE, z DOUBLE')

diamonds = spark.read.option('delimiter', '\t').option('header', True).schema(diamonds_schema).csv('/FileStore/tables/diamonds.txt')

diamonds.printSchema()

# COMMAND ----------

diamonds.count()

# COMMAND ----------

diamonds.show(10)

# COMMAND ----------

sample_pdf = diamonds.sample(withReplacement= False, fraction = 0.25, seed = 1).toPandas()

plt.scatter(sample_pdf.carat, sample_pdf.price, alpha = 0.5, c = 'orange')
plt.xlabel('Carat')
plt.ylabel('Price')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 4: Diamonds Data (Part 2)

# COMMAND ----------

diamonds.sort('price', ascending = False).show(5)

# COMMAND ----------

diamonds.sort('carat', ascending = False).show(5)

# COMMAND ----------

import pyspark.sql.functions as F

diamonds_ppc = diamonds.withColumn('price_per_carat', F.round(expr('price / carat'), 2))

diamonds_ppc.sort('price_per_carat', ascending = False).show(5)

# COMMAND ----------

diamonds_ppc.sort('price_per_carat').show(5)

# COMMAND ----------

ppc_sample_pdf = diamonds_ppc.sample(withReplacement= False, fraction = 0.25, seed = 1).toPandas()

plt.scatter(ppc_sample_pdf.carat, ppc_sample_pdf.price_per_carat, alpha = 0.5, c = 'lightgreen')
plt.xlabel('Carat')
plt.ylabel('Price per Carat')

plt.show()
