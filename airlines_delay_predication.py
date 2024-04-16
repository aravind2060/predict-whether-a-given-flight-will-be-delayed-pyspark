from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def create_spark_session(app_name="Flight Delay Prediction"):
    """
    This function sets up a Spark session, which is the starting point for working with Spark.
    Spark is a software tool that can help us analyze and process large amounts of data.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, data_path):
    """
    This function reads the Airlines dataset from the specified file path and returns it as a Spark DataFrame.
    A Spark DataFrame is like a spreadsheet, where each row represents a flight and each column represents a piece of information about that flight.
    """
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    return df

def clean_data(df):
    """
    This function cleans the dataset by removing any duplicate flights and filling in any missing information.
    Cleaning the data helps to ensure that the machine learning model we build later on will work as well as possible.
    """
    df = df.dropDuplicates().na.drop()
    return df

def encode_categorical_variables(df):
    """
    This function takes the dataset and converts the airline, origin airport, and destination airport information from text into numbers.
    This is necessary because machine learning models work better with numbers than with text.
    """
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_Indexed").fit(df) for col in ["Airline", "AirportFrom", "AirportTo"]]
    for indexer in indexers:
        df = indexer.transform(df)
    return df

def assemble_features(df, input_cols, output_col):
    """
    This function takes the different pieces of information about each flight (like the day of the week, departure time, and flight duration) and combines them into a single column called "features".
    The machine learning model will use this "features" column to predict whether a flight will be delayed or not.
    """
    assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
    df = assembler.transform(df)
    return df

def split_data(df, train_ratio=0.8, seed=1234):
    """
    This function splits the dataset into two parts: a "training" set and a "testing" set.
    The training set will be used to build the machine learning model, and the testing set will be used to check how well the model works.
    """
    train_data, test_data = df.randomSplit([train_ratio, 1-train_ratio], seed=seed)
    return train_data, test_data

def train_logistic_regression(train_data, features_col='features', label_col='Delay'):
    """
    This function takes the training data and uses a machine learning algorithm called Logistic Regression to build a model that can predict whether a flight will be delayed or not.
    Logistic Regression is a common algorithm for this type of problem.
    """
    lr = LogisticRegression(featuresCol=features_col, labelCol=label_col)
    lr_model = lr.fit(train_data)
    return lr_model

def evaluate_model(model, test_data, label_col='Delay'):
    """
    This function takes the trained machine learning model and evaluates its performance on the testing data.
    It uses a metric called the "Area Under the Receiver Operating Characteristic (ROC) Curve" to measure how well the model is able to distinguish between delayed and on-time flights.
    It also plots the ROC curve to help visualize the model's performance.
    """
    evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
    auc = evaluator.evaluate(model.transform(test_data))

    # Plot the ROC curve
    fpr, tpr, thresholds = evaluator.evaluate(model.transform(test_data), {evaluator.metricName: "rocCurve"})
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    return auc

def tune_logistic_regression(train_data, test_data, features_col='features', label_col='Delay'):
    """
    This function tunes the hyperparameters of the Logistic Regression model using cross-validation.
    Cross-validation is a technique that helps us find the best set of hyperparameters for the model.
    """
    lr = LogisticRegression(featuresCol=features_col, labelCol=label_col)
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    crossval = CrossValidator(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(labelCol=label_col),
                              numFolds=5)

    lr_model = crossval.fit(train_data)
    return lr_model

def main():
    # Create a Spark session
    spark = create_spark_session()

    # Load the Airlines dataset
    data_path = "/workspaces/predict-whether-a-given-flight-will-be-delayed-pyspark/airlines_dataset.csv"
    df = load_data(spark, data_path)

    # Clean the data
    df = clean_data(df)

    # Encode the categorical variables
    df = encode_categorical_variables(df)

    # Assemble the features
    df = assemble_features(df, input_cols=["DayOfWeek", "Time", "Length", "Airline_Indexed", "AirportFrom_Indexed", "AirportTo_Indexed"], output_col="features")

    # Split the data into training and testing sets
    train_data, test_data = split_data(df)

    # Train and tune the Logistic Regression model
    lr_model = tune_logistic_regression(train_data, test_data)

    # Evaluate the tuned model
    auc = evaluate_model(lr_model, test_data)
    print(f"Test Area Under ROC: {auc}")

if __name__ == "__main__":
    main()