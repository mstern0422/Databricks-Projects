# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 417 - Homework 04
# MAGIC **Mallory Stern**

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Diamond Data

# COMMAND ----------

diamonds_schema = ('carat DOUBLE, cut STRING, color STRING, clarity STRING, depth DOUBLE, table DOUBLE, price INTEGER, x DOUBlE, y DOUBLE, z DOUBLE')

diamonds = (spark.read.option('delimiter', '\t').option('header', True).schema(diamonds_schema).csv('/FileStore/tables/diamonds.txt'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 1: Grouping By Cut

# COMMAND ----------

def rank_cut(cut):
    if cut == "Fair":
        return 1
    elif cut == "Good":
        return 2
    elif cut == "Very Good":
        return 3
    elif cut == "Premium":
        return 4
    else:
        return 5

spark.udf.register('rank_cut', rank_cut)

# COMMAND ----------

diamonds.groupBy('cut').agg(expr('COUNT(*) AS n_diamonds'), expr('ROUND(AVG(price), 2) AS avg_price'), expr('ROUND(AVG(carat), 2) AS avg_carat'), expr('ROUND(AVG(depth), 2) AS avg_depth'), expr('ROUND(AVG(table), 2) AS avg_table')).orderBy(rank_cut('cut')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 2: Filtering based on Carat Size

# COMMAND ----------

for i in range(5):
    if i == 0:
        print("The number of diamonds with carat size in range [0, 1) is", diamonds.filter(diamonds.carat <= 1).count(), ".")
        i += 1
    elif i == 1:
        print("The number of diamonds with carat size in range [1, 2) is", diamonds.filter((diamonds.carat > 1) & (diamonds.carat <= 2)).count(), ".")
        i += 1
    elif i == 2:
        print("The number of diamonds with carat size in range [2, 3) is", diamonds.filter((diamonds.carat > 2) & (diamonds.carat <= 3)).count(), ".")
        i += 1
    elif i == 3:
        print("The number of diamonds with carat size in range [3, 4) is", diamonds.filter((diamonds.carat > 3) & (diamonds.carat <= 4)).count(), ".")
        i += 1
    elif i == 4:
        print("The number of diamonds with carat size in range [4, 5) is", diamonds.filter((diamonds.carat > 4) & (diamonds.carat <= 5)).count(), ".")
        i += 1
    else:
        print("The number of diamonds with carat size in range [5, 6) is", diamonds.filter((diamonds.carat > 5) & (diamonds.carat <= 6)).count(), ".")
        i += 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 3: Binning by Carat Size

# COMMAND ----------

def CARAT_BIN(carat):
    if carat <= 1.0:
        return "[0,1)"
    elif carat > 1.0 & carat <= 2.0:
        return "[1,2)"
    elif carat > 2.0 & carat <= 3.0:
        return "[2,3)"
    elif carat > 3.0 & carat <= 4.0:
        return "[3,4)"
    elif carat > 4.0 & carat <= 5.0:
        return "[4,5)"
    else:
        return "[5,6)"

spark.udf.register('CARAT_BIN', CARAT_BIN)

# COMMAND ----------

diamonds.select('*', expr('CARAT_BIN(carat) AS carat_bin')).groupBy('carat_bin').agg(expr('COUNT(*) AS n_diamonds'), expr('ROUND(AVG(price), 2) AS avg_price')).orderBy('carat_bin').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load IMBD Data

# COMMAND ----------

movies = (spark.read.option('delimiter', '\t').option('header', True).option('inferSchema', True).csv('/FileStore/tables/imdb/movies.txt'))

movies.printSchema()

# COMMAND ----------

names = (spark.read.option('delimiter', '\t').option('header', True).option('inferSchema', True).csv('/FileStore/tables/imdb/names.txt'))

names.printSchema()

# COMMAND ----------

title_principals = (spark.read.option('delimiter', '\t').option('header', True).option('inferSchema', True).csv('/FileStore/tables/imdb/title_principals-1.txt'))

title_principals.printSchema()

# COMMAND ----------

ratings = (spark.read.option('delimiter', '\t').option('header', True).option('inferSchema', True).csv('/FileStore/tables/imdb/ratings-1.txt'))

ratings.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 4: Number of Appearances by Actor

# COMMAND ----------

title_principals.filter((title_principals.category == "actor") | (title_principals.category == "actress")).groupBy('imdb_name_id').agg(expr('COUNT(*) AS appearances')).join(other = names, on = 'imdb_name_id', how = 'left').select('name', 'appearances').orderBy('appearances', ascending=False).show(16)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 5: Average Rating by Director

# COMMAND ----------

title_principals.filter(title_principals.category == "director").join(other = ratings, on = 'imdb_title_id', how = 'left').groupBy('imdb_name_id').agg(expr('COUNT(imdb_title_id) AS num_films'), expr('SUM(total_votes) AS total_votes'), expr('ROUND(SUM(rating)/COUNT(rating), 2) AS avg_rating')).filter("total_votes >= 1000000").filter("num_films >= 5").join(other = names, on = 'imdb_name_id', how = 'left').select('name', 'num_films', 'total_votes', 'avg_rating').show(truncate = False, n = 16)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 6: Actors Appearing in Horror Films

# COMMAND ----------

horror_films = movies.filter(expr('genre LIKE "%Horror%"'))

horror_films.count()

# COMMAND ----------

title_principals.filter((title_principals.category == "actor") | (title_principals.category == "actress")).join(other = horror_films, on = 'imdb_title_id', how = 'semi').groupBy('imdb_name_id').agg(expr('COUNT(*) AS num_films')).join(other = names, on = 'imdb_name_id', how = 'left').select('name', 'num_films').orderBy('num_films', ascending = False).show(16)

# COMMAND ----------


