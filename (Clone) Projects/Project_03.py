# Databricks notebook source
# MAGIC %md
# MAGIC # DSCI 417 - Project 03
# MAGIC ## Forest Cover Prediction
# MAGIC **Mallory Stern**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: Set up Environment
# MAGIC In this portion of the project, the required tool needed from pyspark will be imported in addition to a SparkSession object being created.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B: Load and Explore the Data
# MAGIC For this project, the Forest Cover dataset is going to be utilized. A model will be created that generates predictions about the type of forest cover in a particular wilderness region based on cartographic information. The data was collected from the Roosevelt National Forest in Northern Colorado. Each observation within the data represents a 30-meter by 30-meter patch of land. To begin, the dataset will be loaded into a DataFrame.

# COMMAND ----------

forest_schema = ('Elevation INTEGER, Aspect INTEGER, Slope INTEGER, Horizontal_Distance_To_Hydrology INTEGER, Vertical_Distance_To_Hydrology INTEGER, Horizontal_Distance_To_Roadways INTEGER, Hillshade_9am INTEGER, Hillshade_Noon INTEGER, Hillshade_3pm INTEGER, Horizontal_Distance_To_Fire_Points INTEGER, Wilderness_Area STRING, Soil_Type INTEGER, Cover_Type INTEGER')

fc = (spark.read.option('delimiter', '\t').option('header', True).schema(forest_schema).csv('/FileStore/tables/forest_cover.txt'))

fc.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Now the first few rows of the DataFrame will be shown. The columns of the DataFrame are too wide to fit on the screen all at once, so they will be broken up into two pieces.

# COMMAND ----------

fc_list = fc.columns

fc.select(fc_list[:6]).show(3)
fc.select(fc_list[6:]).show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC The number of observations in the dataset will now be determined.

# COMMAND ----------

N = fc.count()

print(N)

# COMMAND ----------

# MAGIC %md
# MAGIC The proportions of records in each of the two label categories will now be determined. 

# COMMAND ----------

fc.groupBy('Cover_Type').agg(expr(f'ROUND(COUNT(*) / {N}, 2) AS prop')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C: Preprocessing and Splitting the Data
# MAGIC The stages for the preprocessing pipeline will be created in the following segment. Cover_Type will be used as the label the models, and all other columns will be used as features. There will be two vector assemblers in the pipeline. One will create a features vector for logistic regression, and the other will create a features vector to be used for decision trees. 

# COMMAND ----------

cat_features = ['Wilderness_Area', 'Soil_Type']
num_features = [n for n in fc.columns[:-1] if n not in cat_features]

ix_features = [c + '_ix' for c in cat_features]
vec_features = [c + '_vec' for c in cat_features]

indexer = StringIndexer(inputCols = cat_features, outputCols = ix_features)

encoder = OneHotEncoder(inputCols = ix_features, outputCols = vec_features,dropLast = False)

assembler_lr = VectorAssembler(inputCols = num_features + vec_features, outputCol = 'features_lr')
assembler_dt = VectorAssembler(inputCols = num_features + ix_features, outputCol = 'features_dt')

# COMMAND ----------

# MAGIC %md
# MAGIC A pipeline will be created from the stages above and then applied to the data. 

# COMMAND ----------

pipeline = Pipeline(stages=[indexer, encoder, assembler_lr, assembler_dt]).fit(fc)
fc_proc = pipeline.transform(fc)
fc_proc.persist()

fc_proc.select('features_dt', 'Cover_Type').show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC In order to better assess the final model's performance on out-of-sample data, 20% of the dataset will be set aside to form a test set. The remaining 80% will be used to form a training set that will be used to train the models.

# COMMAND ----------

splits = fc_proc.randomSplit([0.8, 0.2], seed=1)
train = splits[0]
test = splits[1]

train.persist()

print('Training Observations:', train.count())
print('Testing Observations:', test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part D: Hyperparameter for Logistic Regression
# MAGIC An accuracy evaluator will be created to be used when hyperparameter tuning.

# COMMAND ----------

accuracy_eval = MulticlassClassificationEvaluator(labelCol = "Cover_Type", predictionCol = "prediction", metricName = "accuracy")

# COMMAND ----------

# MAGIC %md
# MAGIC Grid search and cross-validation will be used to perform hyperparameter tuning for logistic regression.

# COMMAND ----------

lr = LogisticRegression(labelCol = "Cover_Type", featuresCol = "features_lr")

param_grid = (ParamGridBuilder().addGrid(lr.regParam, [0.00001, 0.0001, 0.001, 0.01, 0.1]).addGrid(lr.elasticNetParam, [0, 0.5, 1])).build()

cv = CrossValidator(estimator = lr, estimatorParamMaps = param_grid, evaluator = accuracy_eval, numFolds = 5, seed = 1, parallelism = 1)

cv_model = cv.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC The optimal model found by the grid search algorithm will now be identified. 

# COMMAND ----------

lr_model = cv_model.bestModel

lr_regParam = cv_model.bestModel.getRegParam()
lr_enetParam = cv_model.bestModel.getElasticNetParam()

print("Max CV Score:", max(cv_model.avgMetrics))
print("Optimal Lambda:", lr_regParam)
print("Optimal Alpha:", lr_enetParam)

# COMMAND ----------

# MAGIC %md
# MAGIC A plot will generated to display the results of the cross-validation.

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
# MAGIC ## Part E: Hyperparameter Tuning for Decision Trees
# MAGIC Grid search and cross-validation will be used to perform hyperparameter tuning for decision trees.

# COMMAND ----------

dtree = DecisionTreeClassifier(featuresCol = 'features_dt', labelCol = 'Cover_Type', seed = 1, maxBins = 38)

param_grid = (ParamGridBuilder().addGrid(dtree.maxDepth, [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24] ).addGrid(dtree.minInstancesPerNode, [1, 2, 4])).build()

cv_dt = CrossValidator(estimator = dtree, estimatorParamMaps = param_grid, numFolds = 5, evaluator = accuracy_eval, seed = 1)

cv_model = cv_dt.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC The optimal model found by the grid search algorithm will now be identified. 

# COMMAND ----------

dt_model = cv_model.bestModel
model_maxDepth = dt_model.getMaxDepth()
model_minInstances = dt_model.getMinInstancesPerNode()

print("Max CV Score:", round(max(cv_model.avgMetrics),4))
print("Optimal Depth:   ", model_maxDepth)
print("Optimal MinInst:  ", model_minInstances)

# COMMAND ----------

# MAGIC %md
# MAGIC A plot will generated to display the results of the cross-validation.

# COMMAND ----------

model_params = cv_model.getEstimatorParamMaps()
dt_cv_summary_list = []
for param_set, acc in zip(model_params, cv_model.avgMetrics):
    new_set = list(param_set.values()) + [acc]
    dt_cv_summary_list.append(new_set)
cv_summary = pd.DataFrame(dt_cv_summary_list,
columns=['maxDepth', 'minInst', 'acc'])
for mi in cv_summary.minInst.unique():
    sel = cv_summary.minInst == mi
    plt.plot(cv_summary.maxDepth[sel], cv_summary.acc[sel], label=mi)
    plt.scatter(cv_summary.maxDepth[sel], cv_summary.acc[sel])
plt.legend()
plt.grid()
plt.xticks(range(4,26,2))
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Score')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The feature importance for each of the features used in the optimal decision tree model will be displayed.

# COMMAND ----------

features = num_features + cat_features

pd.DataFrame({
    'feature':features,
    'importance':dt_model.featureImportances
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part F: Identifying and Evaluating the Final Model
# MAGIC Previously, an optimal (or near-optimal) logistic regression model named lr_model and decision tree model named dt_model have been identified. Which of these two models appears to have better performance on out-of-sample observations will be decided based off of their cross-validation scores. The lr_model had a score of around 0.70, while the dt_model got a score of 0.78. Since the dt_model recieved a higher score, we will select this for the final model. This model will be used to generate predictions from the test set.

# COMMAND ----------

test_pred = dt_model.transform(test)
test_pred.select("probability", "prediction", "Cover_Type").show(10, truncate = False)

# COMMAND ----------

# MAGIC %md
# MAGIC These test predictions will be used to evaluate the final model's performance on out-of-sample data.

# COMMAND ----------

pred_and_labels = test_pred.rdd.map(lambda x:(x['prediction'],float(x['Cover_Type'])))
metrics = MulticlassMetrics(pred_and_labels)

print('Test Set Accuracy: ', round(metrics.accuracy, 4))

# COMMAND ----------

# MAGIC %md
# MAGIC The confusion matrix for the test data will be displayed.

# COMMAND ----------

Cover_Type = [1, 2, 3, 4, 5, 6, 7]

cm = metrics.confusionMatrix().toArray().astype(int)

df = pd.DataFrame(
    data=cm, 
    columns=Cover_Type,
    index=Cover_Type
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Observations in the test set with Cover Type 1 were misclassified by the model as Cover Type 2 a total of 101 times. This was the most common type of misclassification in the test set. 

# COMMAND ----------

# MAGIC %md
# MAGIC Now, the precision and recall for all 7 label classes will be displayed.

# COMMAND ----------

print('Label  Precision   Recall')
print('-------------------------')
i = 1

while i < 8:
    print(i, '    ', round(metrics.precision(i), 4), '    ', round(metrics.recall(i), 4))
    i += 1

# COMMAND ----------

# MAGIC %md
# MAGIC According to the metrics given above, the cover type that is most likely to be classified in the final model is cover type 9, with a precision of 0.91 and a recall of 0.93. The cover type most likely to be misclassified would be cover type 2, with a precision score of 0.61 and a recall of 0.54. The cover type with the greatest variance between it's precision and recall scores is again cover type 2, with precision referring to how often the model is correct when predicting the target class, and recall referring to how well the model is able to identify all positive clss instances from the dataset. 
