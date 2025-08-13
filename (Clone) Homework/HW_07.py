# Databricks notebook source
# MAGIC %md
# MAGIC ## DSCI 417 - Homework 07
# MAGIC **Mallory Stern**

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import DecisionTreeClassifier

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Problem 1: Decision Tree Classification

# COMMAND ----------

# MAGIC %md
# MAGIC <pre>
# MAGIC +----+----+----+----+----+-----------+
# MAGIC | x0 | x1 | x2 | x3 | x4 | prediction| Leaf Node
# MAGIC +----+----+----+----+----+-----------+ ---------
# MAGIC | 3.7| 5.6| 3.6| 2.0| 1.0|        0.0|         3
# MAGIC | 8.2| 4.2| 2.1| 2.0| 0.0|        0.0|         7
# MAGIC | 5.4| 3.9| 4.9| 1.0| 1.0|        1.0|         6
# MAGIC | 2.8| 6.1| 8.1| 0.0| 0.0|        2.0|         2
# MAGIC +----+----+----+----+----+-----------+
# MAGIC </pre>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Problem 2: Random Forest Classification

# COMMAND ----------

print("Tree Model 1 Prediction: ", 0.0)
print("Tree Model 2 Prediction: ", 0.0)
print("Tree Model 3 Prediction: ", 1.0)
print("Random Forest Prediction:", 0.0)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ### Problem 3: Load and Process Stroke Data

# COMMAND ----------

stroke_schema = ('gender STRING, age DOUBLE, hypertension INTEGER, heart_disease INTEGER, ever_married STRING, work_type STRING, residence_type STRING, avg_glucose_level DOUBLE, bmi DOUBLE, smoking_status STRING, stroke INTEGER')

stroke_df = (spark.read.option('delimiter', ',').option('header', True).schema(stroke_schema).csv('/FileStore/tables/stroke_data.csv'))

stroke_df.printSchema()

# COMMAND ----------

num_features = ['age', 'avg_glucose_level', 'bmi']
cat_features = [c for c in stroke_df.columns[:-1] if c not in num_features]

ix_features = [c + '_ix' for c in cat_features]

indexer = StringIndexer(inputCols = cat_features, outputCols = ix_features)

assembler = VectorAssembler(inputCols=num_features + ix_features, outputCol='features')

# COMMAND ----------

pipeline = Pipeline(stages=[indexer, assembler]).fit(stroke_df)
train = pipeline.transform(stroke_df)

train.persist()
display(train.select('features', 'stroke').limit(10), truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Problem 4: Hyperparameter Tuning for Decision Trees

# COMMAND ----------

accuracy_eval = MulticlassClassificationEvaluator(labelCol = "stroke", predictionCol = "prediction", metricName = "accuracy")

dtree = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'stroke', seed = 1)

param_grid = (ParamGridBuilder().addGrid(dtree.maxDepth, [2, 4, 6, 8, 10, 12, 14, 16] ).addGrid(dtree.minInstancesPerNode, [4, 8, 16, 32])).build()

cv = CrossValidator(estimator = dtree, estimatorParamMaps = param_grid, numFolds = 5, evaluator = accuracy_eval, seed = 1)
cv_model = cv.fit(train)

# COMMAND ----------

model = cv_model.bestModel
model_maxDepth = model.getMaxDepth()
model_minInstances = model.getMinInstancesPerNode()

print("Max CV Score:", round(max(cv_model.avgMetrics),4))
print("Optimal Depth:    ", model_maxDepth)
print("Optimal MinInst: ", model_minInstances)

# COMMAND ----------

model_params = cv_model.getEstimatorParamMaps()
dt_cv_summary_list = []

for param_set, acc in zip(model_params, cv_model.avgMetrics):
    new_set = list(param_set.values()) + [acc]
    dt_cv_summary_list.append(new_set)

cv_summary = pd.DataFrame(
    dt_cv_summary_list,
    columns=['maxDepth', 'minInst', 'acc']
)

for mi in cv_summary.minInst.unique():
    sel = cv_summary.minInst == mi
    plt.plot(cv_summary.maxDepth[sel], cv_summary.acc[sel], label=mi)
    plt.scatter(cv_summary.maxDepth[sel], cv_summary.acc[sel])

plt.legend()
plt.grid()
plt.xticks(range(2, 18, 2))
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Score')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Problem 5: Structure of Final Model

# COMMAND ----------

print(model.toDebugString)

# COMMAND ----------

features = num_features + cat_features
print(features)

# COMMAND ----------

print("First Feature Used in Tree:", features[0])
print("Features Unused in Tree:", features[6], ",", features[7], ",", features[8])

# COMMAND ----------

pd.DataFrame({
    'feature':features,
    'importance':model.featureImportances
})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Problem 6: Applying the Model to New Data

# COMMAND ----------

new_schema = ('gender STRING, age DOUBLE, hypertension INTEGER, heart_disease INTEGER, ever_married STRING, work_type STRING, residence_type STRING, avg_glucose_level DOUBLE, bmi DOUBLE, smoking_status STRING')

new_data_list = [
    ['Female', 42.0, 1, 0, 'No', 'Private', 'Urban', 182.1, 26.8, 'smokes'],
    ['Female', 64.0, 1, 1, 'Yes', 'Self-employed', 'Rural', 171.5, 32.5, 'formerly smoked'],
    ['Male', 37.0, 0, 0, 'Yes', 'Private', 'Rural', 79.2, 18.4, 'Unknown'],
    ['Male', 72.0, 0, 1, 'No', 'Govt_job', 'Urban', 125.7, 19.4, 'never smoked']
]

new_data = spark.createDataFrame(new_data_list, schema = new_schema)

new_data.show()

# COMMAND ----------

new_train = pipeline.transform(new_data)
new_train.persist()
new_predictions = model.transform(new_train)
display(new_predictions.select("probability", "prediction"))

# COMMAND ----------


