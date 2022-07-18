from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.types import StringType

from pyspark.sql.functions import when
from pyspark.sql.functions import lit
import pyspark.sql.functions as F
from pyspark.sql.functions import repeat, expr

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import json
import time


spark = SparkSession.builder \
    .master("local[10]") \
    .appName("Assignment part1") \
    .config("spark.local.dir","/fastdata/acq21rl") \
    .config("spark.sql.debug.maxToStringFields",2000) \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")  

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


#Load dataset and preprocessing
all_train_data = spark.read.csv('../Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv')
all_test_data = spark.read.csv('../Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv')

all_train_data.cache()
all_test_data.cache()

ncolumns = len(all_train_data.columns)
col_names = all_train_data.schema.names

# rename the label column's name
all_train_data = all_train_data.withColumnRenamed('_c128','labels')
all_test_data = all_test_data.withColumnRenamed('_c128','labels')

# transform the data typer from string to double to feed the model
StringColumns = [x.name for x in all_train_data.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    all_train_data = all_train_data.withColumn(c, col(c).cast("double"))
    all_test_data = all_test_data.withColumn(c, col(c).cast("double"))

# transform the label from -1 to 0, because the model's output only range over 0
all_train_data = all_train_data.withColumn('labels',when(col('labels')==-1,0).otherwise(all_train_data.labels))
all_test_data = all_test_data.withColumn('labels',when(col('labels')==-1,0).otherwise(all_test_data.labels))
all_train_data.cache()
all_test_data.cache()

# Subset of the trian dataset
data = all_train_data.sample(0.01,1500)
train,test = data.randomSplit([0.7,0.3],123)
train.cache()
test.cache()
print("The number of subset train and val dataset is:",data.count())

# combine the features into one column
vecAssembler = VectorAssembler(inputCols = col_names[0:ncolumns-1], outputCol = 'features')
vec_train_data = vecAssembler.transform(train)
vec_train_data.select("features", "labels").show(5)

# get the accuracy evaluation
evaluator1 = MulticlassClassificationEvaluator\
    (labelCol="labels", predictionCol="prediction", metricName="accuracy")

# get the area under the curve evaluation
evaluator2 = BinaryClassificationEvaluator\
    (labelCol="labels", metricName="areaUnderROC")




def train_rf(tr_data,te_data):
  rf = RandomForestClassifier(labelCol="labels", featuresCol="features", seed=42)
  stages_rf = [vecAssembler, rf]
  pipeline_rf = Pipeline(stages=stages_rf)
  
  # make the parameters grid
  paramGrid_rf = ParamGridBuilder() \
      .addGrid(rf.maxDepth, [5, 6, 7]) \
      .addGrid(rf.maxBins, [2, 3, 4]) \
      .addGrid(rf.subsamplingRate, [0.1, 0.5,0.8]) \
      .build()
  
  # set the cross validator
  crossval_rf = CrossValidator(estimator=pipeline_rf,
                        estimatorParamMaps=paramGrid_rf,
                        evaluator=evaluator1,
                        numFolds=3)
  
  print("**********start training rf model***************")
  
  cvModel_rf = crossval_rf.fit(tr_data) 
  prediction_rf = cvModel_rf.transform(te_data)
  accuracy_rf = evaluator1.evaluate(prediction_rf)
  
  print("Accuracy for best rf model = %g " % accuracy_rf)
  
  paramDict_rf = {param[0].name: param[1] for param in cvModel_rf.bestModel.stages[-1].extractParamMap().items()}
  print(json.dumps(paramDict_rf, indent = 4))
  
  # save the best model with the best paras after train the model to reuse in the whole data
  best_model = cvModel_rf.bestModel
  return best_model



## retrain the rf model
def retrain_rf(tr_data,te_data,best_model):

  # get the best params from the best model we trained before
  best_maxDepth = best_model.stages[-1]._java_obj.parent().getMaxDepth()
  best_maxBins = best_model.stages[-1]._java_obj.parent().getMaxBins()
  best_subsamplingRate = best_model.stages[-1]._java_obj.parent().getSubsamplingRate()
  
  print("The trained best maxDepth:",best_maxDepth)
  print("The trained best maxBins:",best_maxBins)
  print("The trained best subsamplingRate:",best_subsamplingRate)
  
  rf = RandomForestClassifier(labelCol="labels", featuresCol="features",maxDepth = best_maxDepth, maxBins =best_maxBins,subsamplingRate = best_subsamplingRate)
  stages_rf = [vecAssembler, rf]
  pipeline_rf = Pipeline(stages=stages_rf)
  
  model = pipeline_rf.fit(tr_data)
  prediction = model.transform(te_data)
  accuracy_rf = evaluator1.evaluate(prediction)
  auc_rf = evaluator2.evaluate(prediction)
  
  print("Accuracy for rf model in whole dataset = %g " % accuracy_rf)
  print("AUC for rf model in whole dataset = %g " % auc_rf)




def train_lr(tr_data,te_data):
  lr = LogisticRegression(featuresCol='features', labelCol='labels',threshold = 0)
  stages_lr = [vecAssembler, lr]
  pipeline_lr = Pipeline(stages=stages_lr)
  
  # make the parameters grid
  paramGrid_lr = ParamGridBuilder() \
      .addGrid(lr.elasticNetParam,[0,0.5,1]) \
      .addGrid(lr.regParam,[0.01,0.001,0.0001]) \
      .addGrid(lr.maxIter,[100,200,400]) \
      .build()
  
  # set the cross validator  
  crossval_lr = CrossValidator(estimator=pipeline_lr,
                        estimatorParamMaps=paramGrid_lr,
                        evaluator=evaluator1,
                        numFolds=5)
                        
  print("**********start training lr model***************")
  cvModel_lr = crossval_lr.fit(tr_data)
  prediction_lr = cvModel_lr.transform(te_data)
  accuracy_lr = evaluator1.evaluate(prediction_lr)

  print("Accuracy for best lr model = %g " % accuracy_lr)
  
  paramDict = {param[0].name: param[1] for param in cvModel_lr.bestModel.stages[-1].extractParamMap().items()}
  print(json.dumps(paramDict, indent = 4))
  
  # save the best model with the best paras after train the model to reuse in the whole data
  best_model = cvModel_lr.bestModel
  return best_model



## retrain the lr model
def retrain_lr(tr_data,te_data,best_model):

  # get the best params from the best model we trained before
  best_elasticNetParam = best_model.stages[-1]._java_obj.parent().getElasticNetParam()
  best_regParam = best_model.stages[-1]._java_obj.parent().getRegParam()
  best_maxIter = best_model.stages[-1]._java_obj.parent().getMaxIter()
  
  print("The trained best elasticNetParam:",best_elasticNetParam)
  print("The trained best regParam:",best_regParam)
  print("The trained best maxIter:",best_maxIter)
  
  lr = LogisticRegression(featuresCol='features', labelCol='labels',elasticNetParam = best_elasticNetParam, regParam = best_regParam,maxIter = best_maxIter)
  stages_lr = [vecAssembler, lr]
  pipeline_lr = Pipeline(stages=stages_lr)
  
  print("**********start training lr model on whole dataset***************")
  cvModel_lr = pipeline_lr.fit(tr_data)
  prediction_lr = cvModel_lr.transform(te_data)
  accuracy_lr = evaluator1.evaluate(prediction_lr)
  auc_lr = evaluator2.evaluate(prediction_lr)
  
  print("Accuracy for lr model on the whole dataset = %g " % accuracy_lr)
  print("AUC for lr model on the whole dataset = %g " % auc_lr)
  
  
  
def train_MLP(data,test):
  mlp = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", layers = [128,32,2], seed=1500)
  stages = [vecAssembler, mlp]
  pipeline = Pipeline(stages=stages)
  
  # make the parameters grid
  paramGrid_mlp = ParamGridBuilder() \
    .addGrid(mlp.blockSize,[128,256,512]) \
    .addGrid(mlp.maxIter,[100,200,300]) \
    .build()

  
  # set the cross validator  
  crossval_mlp = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid_mlp,
                        evaluator=evaluator1,
                        numFolds=3)
                        
  print("**********start training MLP model***************")    
  pipelineModel = crossval_mlp.fit(data)
  predictions = pipelineModel.transform(data)
  accuracy = evaluator1.evaluate(predictions)
  
  print("Accuracy of MLP = %g " % accuracy)
  
  paramDict = {param[0].name: param[1] for param in pipelineModel.bestModel.stages[-1].extractParamMap().items()}
  print(json.dumps(paramDict, indent = 4))
  
  # save the best model with the best paras after train the model to reuse in the whole data
  best_model = pipelineModel.bestModel
  return best_model
  
# retrain the mlp model
def retrain_MLP(all_train_data,all_test_data,best_model):
  # get the best params from the best model we trained before
  best_blockSize = best_model.stages[-1]._java_obj.parent().getBlockSize()
  best_maxIter = best_model.stages[-1]._java_obj.parent().getMaxIter()
  
  print("The trained best blockSize of mlp:",best_blockSize)
  print("The trained best maxIter of mlp:",best_maxIter)
  
  mlp = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features",blockSize = best_blockSize, maxIter=best_maxIter, layers=[128,32,2], seed=1500)
  stages = [vecAssembler, mlp]
  pipeline = Pipeline(stages=stages)
  
  print("**********start training MLP model***************")    
  pipelineModel = pipeline.fit(all_train_data)
  predictions = pipelineModel.transform(all_test_data)
  accuracy = evaluator1.evaluate(predictions)
  auc = evaluator2.evaluate(predictions)
  
  print("Accuracy of MLP = %g " % accuracy)
  print("AUC of MLP = %g " % auc)

# the main part of the project to show the procedures
print("****Now start to train the 1% data*****")
start = time.time()
best_model_rf = train_rf(train,test)
rf = time.time()
print("The running time of Random Forest for 1% dataset is:",rf-start)
best_model_lr = train_lr(train,test)
lr = time.time()
print("The running time of linear regression for 1% dataset is:",lr-rf)
best_model_MLP = train_MLP(train,test)
mlp = time.time()
print("The running time of MLP for 1% dataset is:",mlp-lr)
print("The total using time:",mlp-start)


print("****Now start to train the whole data*****")
start2 = time.time()
retrain_rf(all_train_data,all_test_data,best_model_rf)
rf2 = time.time()
print("The running time of Random Forest for 100% dataset is:",rf2-start2)
retrain_lr(all_train_data,all_test_data,best_model_lr)
lr2 = time.time()
print("The running time of linear regression for 100% dataset is:",lr2-rf2)
retrain_MLP(all_train_data,all_test_data,best_model_MLP)
mlp2 = time.time()
print("The running time of MLP for 100% dataset is:",mlp2-lr2)

spark.stop()