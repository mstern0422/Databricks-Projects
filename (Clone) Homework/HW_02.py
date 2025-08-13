# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 417 - Homework 02
# MAGIC **Mallory Stern**

# COMMAND ----------

from pyspark.sql import SparkSession
import pandas as pd
from string import punctuation

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 1: Word Count

# COMMAND ----------

ws_lines = sc.textFile('/FileStore/tables/shakespeare_complete.txt')

ws_words = ws_lines.flatMap(lambda x : x.split(' ')).flatMap(lambda x : x.split('-')).flatMap(lambda x : x.split('_')).flatMap(lambda x : x.split('.')).flatMap(lambda x : x.split(',')).flatMap(lambda x : x.split(':')).flatMap(lambda x : x.split('|')).flatMap(lambda x : x.split('\t')).map(lambda x: x.strip(punctuation)).map(lambda x: x.strip("0123456789")).map(lambda x: x.replace("'", '')).map(lambda x: x.lower()).filter(lambda x: x not in '')

dist_words = ws_words.distinct()

print("Total Number of Words:    ", ws_words.count())
print("Number of Distinct Words: ", dist_words.count())

# COMMAND ----------

ws_words_sample = ws_words.sample(withReplacement = False, fraction = 0.0001)
print(ws_words_sample.collect())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 2: Longest Words

# COMMAND ----------

def longer_str(str1, str2):
    if len(str1) > len(str2):
        return str1
    elif len(str1) < len(str2):
        return str2
    else:
        if str1 > str2:
            return str2
        else:
            return str1
        
print(dist_words.reduce(longer_str))

# COMMAND ----------

print(dist_words.sortBy(len, ascending = False).take(20))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 3: Word Frequency

# COMMAND ----------

pair = ws_words.map(lambda x: (x, 1))

word_counts = pair.reduceByKey(lambda x, y : x + y).sortBy(lambda x: x[1], ascending = False)
word_counts_list = word_counts.take(20)

df = pd.DataFrame(word_counts_list, columns = ['Word', 'Count'])
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 4: Removing Stop Words

# COMMAND ----------

sw_rdd = sc.textFile('/FileStore/tables/stopwords.txt')
print(sw_rdd.count())
print(sw_rdd.sample(withReplacement = False, fraction = 0.05).collect())
sw = sw_rdd.collect()

# COMMAND ----------

ws_words_f = ws_words.filter(lambda x: x not in sw)
dist_words_f = ws_words_f.distinct()

print("Number of Distinct Non-Stop Words:", dist_words_f.count())

# COMMAND ----------

pair = ws_words_f.map(lambda x: (x, 1))

word_counts = pair.reduceByKey(lambda x, y : x + y).sortBy(lambda x: x[1], ascending = False)
word_counts_list = word_counts.take(20)

df = pd.DataFrame(word_counts_list, columns = ['Word', 'Count'])
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 5: Diamonds Dataset

# COMMAND ----------

diamonds_raw = sc.textFile('/FileStore/tables/diamonds.txt')
print(diamonds_raw.count())

# COMMAND ----------

for row in diamonds_raw.take(5):
    print(row)

# COMMAND ----------

def process_row(row):
    item = row.split('\t')
    return [float(item[0]), str(item[1]), str(item[2]), str(item[3]), float(item[4]), float(item[5]), int(item[6]), float(item[7]), float(item[8]), float(item[9])]

diamonds = diamonds_raw.filter(lambda x: 'carat' not in x).map(process_row)

for row in diamonds.take(5):
    print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 6: Grouped Means

# COMMAND ----------

cut_summary = diamonds.map(lambda row: (row[1], (row[0], row[6], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])).map(lambda x: (x[0], x[1][2], round(x[1][0] / x[1][2], 2), round(x[1][1] / x[1][2], 2))).collect()

cut_df = pd.DataFrame(cut_summary, columns = ['Cut', 'Count', 'Mean_Carat', 'Mean_Price'])
cut_df
