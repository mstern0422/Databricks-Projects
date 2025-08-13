# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 417 - Homework 06
# MAGIC **Mallory Stern**

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 1: Load Stroke Data

# COMMAND ----------

stroke_schema = ('gender STRING, age DOUBLE, hypertension INTEGER, heart_disease INTEGER, ever_married STRING, work_type STRING, residence_type STRING, avg_glucose_level DOUBLE, bmi DOUBLE, smoking_status STRING, stroke INTEGER')

stroke_df = (spark.read.option('delimiter', ',').option('header', True).schema(stroke_schema).csv('/FileStore/tables/stroke_data.csv'))

stroke_df.printSchema()

# COMMAND ----------

stroke_df.show(10)

# COMMAND ----------

N = stroke_df.count()

print(N)

# COMMAND ----------

stroke_df.groupBy('stroke').agg(expr('ROUND(COUNT(*) / {}, 4) AS prop'.format(N))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 2: Preprocessing

# COMMAND ----------

num_features = ['age', 'avg_glucose_level', 'bmi']
cat_features = [c for c in stroke_df.columns[:-1] if c not in num_features]
ix_features = [c + '_ix' for c in cat_features]
vec_features = [c + '_vec' for c in cat_features]

indexer = StringIndexer(inputCols = cat_features, outputCols = ix_features)

encoder = OneHotEncoder(inputCols = ix_features, outputCols = vec_features,dropLast=False)

assembler = VectorAssembler(inputCols = num_features + vec_features, outputCol='features')

# COMMAND ----------

pipeline = Pipeline(stages=[indexer, encoder, assembler]).fit(stroke_df)
train = pipeline.transform(stroke_df)

train.persist()
display(train.select('features', 'stroke').limit(10), truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 3: Hyperparameter Tuning for Logistic Regression

# COMMAND ----------

accuracy_eval = MulticlassClassificationEvaluator(labelCol = "stroke", predictionCol = "prediction", metricName = "accuracy")
lr = LogisticRegression(labelCol="stroke", featuresCol="features")

param_grid = (ParamGridBuilder().addGrid(lr.regParam, [0.0001, 0.001, 0.01, 0.1, 1]).addGrid(lr.elasticNetParam, [0, 0.5, 1])).build()

cv = CrossValidator(estimator = lr, estimatorParamMaps = param_grid, evaluator = accuracy_eval, numFolds = 5, seed = 1, parallelism = 1)

cv_model = cv.fit(train)

# COMMAND ----------

opt_model = cv_model.bestModel

opt_regParam = cv_model.bestModel.getRegParam()
opt_enetParam = cv_model.bestModel.getElasticNetParam()

print("Max CV Score:", max(cv_model.avgMetrics))
print("Optimal Lambda:", opt_regParam)
print("Optimal Alpha:", opt_enetParam)

# COMMAND ----------

model_params = cv_model.getEstimatorParamMaps()
lr_cv_summary_list = []
for param_set, acc in zip(model_params, cv_model.avgMetrics):
    new_set = list(param_set.values()) + [acc]
    lr_cv_summary_list.append(new_set)
cv_summary = pd.DataFrame(lr_cv_summary_list,
                          columns=['reg_param', 'enet_param', 'acc'])
for en in cv_summary.enet_param.unique():
    sel = cv_summary.enet_param == en
    plt.plot(cv_summary.reg_param[sel], cv_summary.acc[sel], label=en)
    plt.scatter(cv_summary.reg_param[sel], cv_summary.acc[sel])
plt.legend()
plt.xscale('log')
plt.grid()
plt.xlabel('Regularization Parameter')
plt.ylabel('Cross-Validation Score')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 4: Training Predictions

# COMMAND ----------

train_pred = opt_model.transform(train)

train_pred.select('probability', 'prediction', 'stroke').show(10, truncate=False)

# COMMAND ----------

train_pred.filter(train_pred.prediction != train_pred.stroke).select('probability', 'prediction', 'stroke').show(10, truncate=False)

# COMMAND ----------

print("The highest probability observed for an incorrect answer is 0.7013.")
print("The lowest probability observed for an incorrect answer is 0.2927")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 5: Classification Metrics

# COMMAND ----------

pred_and_labels = train_pred.rdd.map(lambda x:(x['prediction'],float(x['stroke'])))

# COMMAND ----------

metrics = MulticlassMetrics(pred_and_labels)

print(metrics.accuracy)

# COMMAND ----------

confusion_matrix = metrics.confusionMatrix().toArray()

confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=['0', '1'], index=['0', '1'])

display(confusion_matrix_df)

# COMMAND ----------

precision_0 = round(metrics.precision(0), 4)
recall_0 = round(metrics.recall(0), 4)

precision_1 = round(metrics.precision(1), 4)
recall_1 = round(metrics.recall(1), 4)

print("Label   Precision   Recall")
print("--------------------------")
print("0      ", precision_0,"    ", recall_0)
print("1      ", precision_1,"    ", recall_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem 6: Applying the Model to New Data

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
new_pred = opt_model.transform(new_train)

new_pred.select('probability', 'prediction').show(truncate=False)
