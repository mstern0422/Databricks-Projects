# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 417 - Homework 05
# MAGIC **Mallory Stern**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 1: Sparse Vector Representation

# COMMAND ----------

# MAGIC %md
# MAGIC 1. `[8, [3, 4], [1, 1]]`
# MAGIC 2. `[10, [2, 4, 8], [4, 7, 2]]`
# MAGIC 3. `[10, [2, 8], [1, 1]]`
# MAGIC 4. `[3, [0, 1, 2], [4, 1, 8]]`
# MAGIC 5. `[6, [], []]`

# COMMAND ----------

# MAGIC %md
# MAGIC 1. `[0, 0, 1, 0, 0, 3, 0, 9]`
# MAGIC 2. `[0, 0, 0, 0]`
# MAGIC 3. `[0, 2, 3, 1]`
# MAGIC 4. `[0, 1, 0, 0, 0, 1, 0, 0]`
# MAGIC 5. `[0, 0, 0, 0, 2, 0]`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 2: One-Hot Encoding

# COMMAND ----------

# MAGIC %md
# MAGIC <pre>
# MAGIC +-----+--------+--------------+
# MAGIC |Group|Group_ix|     Group_vec|
# MAGIC +-----+--------+--------------+
# MAGIC |    C|       1| [8, [1], [1]]|
# MAGIC |    E|       2| [8, [2], [1]]|
# MAGIC |    G|       4| [8, [4], [1]]|
# MAGIC |    F|       0| [8, [0], [1]]|
# MAGIC |    A|       6| [8, [6], [1]]|
# MAGIC |    D|       7| [8, [7], [1]]|
# MAGIC +-----+--------+--------------+
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 3: Vector Assembler (Part 1)

# COMMAND ----------

# MAGIC %md
# MAGIC <pre>
# MAGIC +---+---+---+--------------------+
# MAGIC | c1| c2| c3|           features |
# MAGIC +---+---+---+--------------------+
# MAGIC |  3|  B|  Y| |
# MAGIC |  6|  C|  X| |
# MAGIC |  2|  D|  X| |
# MAGIC |  5|  A|  Y| |
# MAGIC |  1|  A|  Z| |
# MAGIC +---+---+---+--------------------+
