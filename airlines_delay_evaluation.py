from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def create_spark_session(app_name="Flight Delay Prediction"):
    """
    This function sets up a Spark session, which is the starting point for working with Spark.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, data_path):
    """
    This function reads the Airlines dataset from the specified file path and returns it as a Spark DataFrame.
    """
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    return df

def clean_data(df):
    """
    This function cleans the dataset by removing any duplicate flights and filling in any missing information.
    """
    df = df.dropDuplicates().na.drop()
    return df

def encode_categorical_variables(df):
    """
    This function takes the dataset and converts the airline, origin airport, and destination airport information from text into numbers.
    """
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_Indexed").fit(df) for col in ["Airline", "AirportFrom", "AirportTo"]]
    for indexer in indexers:
        df = indexer.transform(df)
    return df

def assemble_features(df, input_cols, output_col):
    """
    This function takes the different pieces of information about each flight and combines them into a single column called "features".
    """
    assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
    df = assembler.transform(df)
    return df

def split_data(df, train_ratio=0.8, seed=1234):
    """
    This function splits the dataset into training and testing sets.
    """
    train_data, test_data = df.randomSplit([train_ratio, 1-train_ratio], seed=seed)
    return train_data, test_data

def train_and_evaluate_models(train_data, test_data):
    """
    This function trains and evaluates multiple machine learning models for the flight delay prediction task.
    """
    models = {
        "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="Delay"),
        "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="Delay", maxBins=300),
        "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="Delay", maxBins=300),
        "Gradient Boosting": GBTClassifier(featuresCol="features", labelCol="Delay", maxBins=300)
    }

    trained_models = {}
    results = {}
    for name, model in models.items():
        trained_model = model.fit(train_data)
        evaluator = BinaryClassificationEvaluator(labelCol="Delay")
        auc = evaluator.evaluate(trained_model.transform(test_data))
        results[name] = auc
        trained_models[name] = trained_model
        print(f"{name} Test Area Under ROC: {auc}")

    return results, trained_models  # Ensure this line is correct


def plot_roc_curves(test_data, models):
    """
    This function uses scikit-learn to plot ROC curves for the models based on their predictions.
    """
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        # Get predictions
        predictions = model.transform(test_data)
        # Convert to pandas DataFrame
        predictions_pd = predictions.select(['probability', 'Delay']).toPandas()
        # Calculate probabilities for the positive class
        probabilities = predictions_pd['probability'].apply(lambda x: x[1])
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(predictions_pd['Delay'], probabilities)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

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

    # Train and evaluate multiple models
    results, trained_models = train_and_evaluate_models(train_data, test_data)

    # Plot the ROC curves
    plot_roc_curves(test_data, trained_models)

if __name__ == "__main__":
    main()