# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 05:39:19 2021

@author: ameneh
"""

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, NaiveBayes, DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Mytest5").getOrCreate()
    data_train=spark.read.csv('train.csv',inferSchema=True, header=True)
    data_train=data_train.drop('PassengerId','Name','Cabin','Ticket')
    #print("the data is:\n")
    #data_train.show()
    #fill missing value age to mean
    print("*********************************************")
    age_mean = data_train.select(["Age"]).toPandas().mean()[0]
    data_train = data_train.na.fill( 30,subset='Age')
    data_train = data_train.na.fill( 1,subset='Embarked')
    #print("the data next fill avg:\n")
   # data_train.show()
    print("**********************************************") 
    #delete null field in Embarked
    data_train=data_train.dropna()
   # print("the data is delete null Embarked:\n")
   # data_train.show()
    print("**********************************************") 
    #convert sex and Embarked
    indexer= StringIndexer(inputCol="Sex", outputCol="Sex_index").fit(data_train)
    data_train = indexer.transform(data_train)
    indexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_index").fit(data_train)
    data_train = indexer.transform(data_train)
    print("conver sex and Embarked")
    data_train.show()
    print("***********************************************")
    #create model
    inputcoll=['Pclass','Sex_index','Age','Fare','Embarked_index']
    outputColl = 'features'
    vector=VectorAssembler(inputCols=inputcoll,outputCol=outputColl)
    data_train = vector.transform(data_train)
    
    indexer_test = StringIndexer(inputCol="Survived", outputCol="label").fit(data_train)
    data_train=data_train.withColumnRenamed('Survived', 'label')
    data_train=data_train.select("features","label")
    print("the train data convert to vector is:\n")
    #data_train.show()
    #data test*******************************************************
    data_test=spark.read.csv('test_of_titanic.csv',inferSchema=True, header=True)
    data_test=data_test.drop('PassengerId','Name','Cabin','Ticket')
    #print("the data is:\n")
    #data_test.show()
    #fill missing value age to mean
    print("*********************************************")
    age_mean = data_test.select(["Age"]).toPandas().mode()
    
    data_test = data_test.na.fill(60,subset='Age')
    data_test = data_test.na.fill( 1,subset='Embarked')
    #data_test = data_test.na.fill( 's',subset='Embarked')
    #print("the data next fill avg:\n")
    data_test.show()
    print("**********************************************") 
    #delete null field in Embarked
    data_test=data_test.dropna()
   # print("the data is delete null Embarked:\n")
   # data_test.show()
    print("**********************************************") 
    #convert sex and Embarked
    indexer= StringIndexer(inputCol="Sex", outputCol="Sex_index").fit(data_test)
    data_test = indexer.transform(data_test)
    indexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_index").fit(data_test)
    data_test = indexer.transform(data_test)
    #print("conver sex and Embarked")
    data_test.show()
   
    print("***********************************************")
    #create model
    inputcoll=['Pclass','Sex_index', 'Age', 'Fare','Embarked_index']
    outputColl = 'features'
    vector=VectorAssembler(inputCols=inputcoll,outputCol=outputColl)
    data_vector_test = vector.transform(data_test)
    #data_vector_test.show()
    #data_vector_test=data_vector_test.select('features')
    indexer_test = StringIndexer(inputCol="Survived", outputCol="label").fit(data_vector_test)
    data_vector_test=data_vector_test.withColumnRenamed('Survived', 'label')
    data_vector_test=data_vector_test.select("features","label")
    print("the train data convert to vector is:\n")
    #data_vector_test.show()
    #end test file*********************************************************
    print("**********************************************")
    #logistic regresion
    #x_train= data_train.randomSplit([0.7,0.3])
    #y_test = data_vector_test.randomSplit([0.7,0.3])
    
    
    lr=LogisticRegression(featuresCol = 'features', maxIter=10)
    lr_model=lr.fit(data_train)
    #********************************
    results=lr_model.transform(data_vector_test)
    
    #results=results.select(['features','prediction','rawPrediction'])
    print("result LogisticRegression in test data is:\n")
    #results.show(10)
    #results=results.withColumnRenamed('prediction', 'label')
    results.show(10)
    
    print("************************************************* **")
    #cross validation
   
    
    Eva = BinaryClassificationEvaluator()
    
    #pipeline = Pipeline(stages=[lr])
    paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).addGrid(lr.elasticNetParam, [0, 1]).build()
    crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=Eva,
                          numFolds=5
                          )
    
    #modelEvaluator=RegressionEvaluator()
    #lg_Model = crossval.fit(results)
    lg_result= Eva.evaluate(results)
    print("the score LogisticRegression is",lg_result)
    #************************************************************
    #NaiveBayes
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    model = nb.fit(data_train)
    result_nb=model.transform(data_vector_test)
    result_nb=result_nb.select(['features','label','rawPrediction'])
   # print("\n result NaiveBayes in test data is:\n")
    #result_nb.show(10)
    #*************************************************************

    paramGrid =(ParamGridBuilder()
               .addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
               .build())
    crossval = CrossValidator(estimator=nb,
                          estimatorParamMaps=paramGrid,
                          evaluator=Eva,
                          numFolds=10
                          )
    
    #modelEvaluator=RegressionEvaluator()
    #lg_Model = crossval.fit(results)
    nb_result= Eva.evaluate(result_nb)
    print("the score NaiveBayes is",nb_result)
    #**************************************************************
    #DecisionTree
    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label')
    dtModel = dt.fit(data_train)
    result_dt = dtModel.transform(data_vector_test)
    result_dt=result_dt.select(['features','label','rawPrediction'])
    print("\n result DecisionTreeClassifier in test data is:")
    result_dt.show(10)
    #*****************************************************
    paramGrid =(ParamGridBuilder()
             .addGrid(dt.maxDepth, [2, 5, 10, 20, 30])
             .addGrid(dt.maxBins, [10, 20, 40, 80, 100])
             .build())
    crossval = CrossValidator(estimator=dt,
                          estimatorParamMaps=paramGrid,
                          evaluator=Eva,
                          numFolds=10
                          )
    
    #modelEvaluator=RegressionEvaluator()
    #lg_Model = crossval.fit(results)
    dt_result= Eva.evaluate(result_dt)
    print("the score DecisionTree is",dt_result)
    
    