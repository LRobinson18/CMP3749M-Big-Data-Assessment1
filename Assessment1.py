# Import all relevant libraries and modules
import pyspark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create instance of pyspark library
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Read in the csv, inferring schema, telling it to use headers from the csv
df = spark.read.csv("nuclear_plants_small_dataset.csv",inferSchema=True,header=True)

# Define each task as a definition for easy testing
# Section 1
# Task 1
def Task1(df):
    # For each column in the table, count the null fields
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

    # .isNull() returns a boolean
    nullStatus = df.where(df["Status"].isNull()).count()
    print("The status column has",nullStatus,"empty fields.")

    # If there are any NULL fields
    if nullStatus != 0:
        print("Dropping records with NULL values...")
        # Drop any records with NULL values 
        df = df.dropna()

# Task 2
def Task2(df):
    # Take the columns of the data set and remove the first one
    columnsList = df.columns
    columnsList.pop(0)

    # For each column in the data set
    for c in columnsList:
        # Create new matplotlib frame and subplots
        plt.figure(num=c)

        # Create dataframe for values
        dfSummary = pd.DataFrame({'Summary': ["Minimum","Maximum","Mean","Median","Mode","Variance"]})
        # For both normal and abnormal values
        types = ["Normal", "Abnormal"]
        for t in types:
            # Select normal/abnormal column
            dfCalculations = df.filter(df.Status==t).select(c).summary().toPandas()

            # Get minimum, maximum, mean, median, mode, and variance values from the summary table
            # Convert them all to floats and round to 4 decimal places
            minValue = round(float(dfCalculations.iloc[3][c]),4)
            maxValue = round(float(dfCalculations.iloc[7][c]),4)
            meanValue = round(float(dfCalculations.iloc[1][c]),4)
            medianValue = round(float(dfCalculations.iloc[5][c]),4)
            modeValue = round(float(df.groupby(c).count().orderBy("count", ascending=False).first()[0]),4)
            varianceValue = round(float(dfCalculations.iloc[5][c]) ** 2,4)

            # Make new dataframe to join onto the previous one
            dfAppend = pd.DataFrame({t: [minValue,maxValue,meanValue,medianValue,modeValue,varianceValue]})
            dfSummary = pd.concat([dfSummary,dfAppend], axis=1)

        # Create table to add to matplotlib window
        cell_text = []
        for row in range(len(dfSummary)):
            cell_text.append(dfSummary.iloc[row])

        plt.subplot(211)
        plt.table(cellText=cell_text, colLabels=dfSummary.columns, loc='center')
        plt.axis('off')
        
        # Create box plot
        plt.subplot(212)
        sn.boxplot(x="Status", y=c, data=df.toPandas())
        

    # Show the matplotlib window with the table and corresponding box-plot
    plt.show()

# Task 3
def Task3(df):
    # For both normal and abnormal values
    types = ["Normal", "Abnormal"]
    for t in types:
        # Create new matplotlib frame and subplots
        plt.figure(num=t)
        # Select normal/abnormal column
        # Convert the pyspark dataframe to a pandas dataframe
        df2 = df.filter(df.Status==t).toPandas()

        # Create correlation matrix with heatmap
        corrMatrix = df2.corr()
        sn.heatmap(corrMatrix, annot=True)

    plt.show()

# Section 2
# Task 4
def Task4(df):
    # Replace Normal with 0 and Abnormal with 1
    df = df.replace("Normal","0").replace("Abnormal","1")
    df = df.withColumn("Status",df.Status.cast('int'))

    # Randomly shuffle the dataset
    # Split the dataset 70% train, 30% test
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    # Print the number of examples
    print("Normal = 0, Abnormal = 1")
    print("Training data")
    trainingData.groupBy("Status").count().show()
    print("Test data")
    testData.groupBy("Status").count().show()

    return trainingData, testData

# Task 5
def Task5(trainingData, testData, df):
    # Create dataframe for values
    dfSummary = pd.DataFrame({'Classifiers': ["Error Rate","Sensitivity","Specificity"]})
    
    # Train Decision Tree
    predictions, evaluator = DecisionTree(trainingData, testData, df)
    # Update table
    dfSummary = AddToTable(dfSummary, predictions, evaluator, "Decision Tree")

    # Train Support Vector Machine
    predictions, evaluator = SVM(trainingData, testData, df)
    # Update table
    dfSummary = AddToTable(dfSummary, predictions, evaluator, "SVM")

    # Train Artificial Neural Network
    predictions, evaluator = ANN(trainingData, testData, df)
    # Update table
    dfSummary = AddToTable(dfSummary, predictions, evaluator, "ANN")

    print(dfSummary)

def DecisionTree(trainingData, testData, df):
    # Code adapted from https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier
    # Take the columns of the data set and remove the first one
    columnsList = df.columns
    columnsList.pop(0)

    # Index labels, adding metadata to the label column
    # Fit on whole dataset to include all labels in index
    labelIndexer = StringIndexer(inputCol="Status", outputCol="indexedStatus")
    # Automatically identify categorical features, and index them
    featureAssembler = VectorAssembler(inputCols=columnsList, outputCol="indexedFeatures")

    # Train a DecisionTree model
    dt = DecisionTreeClassifier(labelCol="indexedStatus", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureAssembler, dt])

    # Train model.  This also runs the indexers
    model = pipeline.fit(trainingData)

    # Make predictions based on the test data
    predictions = model.transform(testData)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexedStatus", predictionCol="prediction", metricName="accuracy")

    treeModel = model.stages[2]
    # summary only
    print(treeModel)

    return predictions, evaluator

def SVM(trainingData, testData, df):
    # Code adapted from https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-support-vector-machine
    # Take the columns of the data set and remove the first one
    columnsList = df.columns
    columnsList.pop(0)

    # Index labels, adding metadata to the label column
    # Fit on whole dataset to include all labels in index
    labelIndexer = StringIndexer(inputCol="Status", outputCol="indexedStatus")
    # Automatically identify categorical features, and index them
    featureAssembler = VectorAssembler(inputCols=columnsList, outputCol="indexedFeatures")

    # Instantiate the base classifier
    lsvc = LinearSVC(labelCol="indexedStatus", featuresCol="indexedFeatures", maxIter=10, regParam=0.1)

    # Chain indexers and SVM in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureAssembler, lsvc])

    # Train model.  This also runs the indexers
    model = pipeline.fit(trainingData)

    # Make predictions based on the test data
    predictions = model.transform(testData)

    # Obtain evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedStatus", predictionCol="prediction", metricName="accuracy")

    return predictions, evaluator

def ANN(trainingData, testData, df):
    # Code adapted from https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier
    # Take the columns of the data set and remove the first one
    columnsList = df.columns
    columnsList.pop(0)

    # Index labels, adding metadata to the label column
    # Fit on whole dataset to include all labels in index
    labelIndexer = StringIndexer(inputCol="Status", outputCol="indexedStatus")
    # Automatically identify categorical features, and index them
    featureAssembler = VectorAssembler(inputCols=columnsList, outputCol="indexedFeatures")

    # specify layers for the neural network:
    # input layer of size 12 (features), two intermediate of size 5 and 4
    # and output of size 1 (classes)
    layers = [12, 5, 4, 2]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(labelCol="indexedStatus", featuresCol="indexedFeatures", maxIter=100, layers=layers, blockSize=128, seed=1234)

    # Chain indexers and SVM in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureAssembler, trainer])

    # Train model.  This also runs the indexers
    model = pipeline.fit(trainingData)

    # Make predictions based on the test data
    predictions = model.transform(testData)

    # obtain evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedStatus", predictionCol="prediction", metricName="accuracy")

    return predictions, evaluator

def Calculate_Err_Sens_Spec(predictions, evaluator):
    # Calculate the error rate
    accuracy = evaluator.evaluate(predictions)
    error_rate = 1-accuracy

    # Calculate true/false positives/negatives
    truePositive = predictions.where((col("prediction")=="1") & (col("indexedStatus")==1)).count()
    trueNegative = predictions.where((col("prediction")=="0") & (col("indexedStatus")==0)).count()

    falsePositive = predictions.where((col("prediction")=="1") & (col("indexedStatus")==0)).count()
    falseNegative = predictions.where((col("prediction")=="0") & (col("indexedStatus")==1)).count()

    # Calculate the sensitivity and specificity
    sensitivity = truePositive / (truePositive+falseNegative)
    specificity = trueNegative / (trueNegative+falsePositive)
    
    return error_rate, sensitivity, specificity

def AddToTable(dfSummary, predictions, evaluator, header):
    # Calculate test error, specificity, sensitivity
    error_rate, sensitivity, specificity = Calculate_Err_Sens_Spec(predictions, evaluator)

    # Make new dataframe to join onto the other one
    dfAppend = pd.DataFrame({header: [error_rate,sensitivity,specificity]})
    dfSummary = pd.concat([dfSummary,dfAppend], axis=1)

    return dfSummary

def Task8(spark):
    # Code adapted from https://spark.apache.org/docs/latest/rdd-programming-guide.html#rdd-operations
    # Create RDD from external Data source
    df = spark.read.csv("nuclear_plants_big_dataset.csv", inferSchema=True,header=True)

    # Convert from dataframe to RDD
    rdd = df.drop("Status").rdd

    # Take the columns of the data set and remove the first one
    columnsList = df.columns
    columnsList.pop(0)

    # Create dataframe for values
    dfSummary = pd.DataFrame({'Features': ["Minumum","Maximum","Mean"]})

    # Calculate min, max and mean using MapReduce
    minValue = rdd.reduce(lambda x,y: np.minimum(x, y))
    maxValue = rdd.reduce(lambda x,y: np.maximum(x, y))
    meanValue = rdd.reduce(lambda x,y: np.mean(np.column_stack([x, y]), axis=1))

    # For each column in the data set
    for n in range(len(columnsList)):
        # Make new dataframe to join onto the previous one
        dfAppend = pd.DataFrame({columnsList[n]: [minValue[n],maxValue[n],meanValue[n]]})
        dfSummary = pd.concat([dfSummary,dfAppend], axis=1)

    print(dfSummary)

#Task1(df)
#Task2(df)
#Task3(df)
trainingData, testData = Task4(df)
Task5(trainingData, testData, df)
#Task8(spark)