# Flight Delay Prediction Using Spark ML

## Overview

This project is focused on predicting flight delays using historical flight data. By leveraging Apache Spark's capabilities in handling large datasets and its machine learning library (Spark ML), we aim to build models that can accurately predict whether a flight will be delayed based on various features such as airline, departure airport, destination airport, day of the week, departure time, and flight duration.

## Prerequisites

To run this project, you will need:
- Apache Spark: A powerful data processing framework that handles big data tasks efficiently.
- Python: The programming language used to write the scripts.
- Libraries: Several Python libraries including `matplotlib` for plotting graphs and `pyspark` for running Spark applications.

Ensure that you have Spark installed and properly configured on your system or server. Python should also be installed, along with the necessary libraries which can be installed using pip:
```bash
pip install matplotlib pyspark
```

## Project Structure

- **Spark Session Setup**: Initializes a Spark session which is necessary to execute all Spark operations.
- **Data Loading**: Loads the flight data from a CSV file into a Spark DataFrame.
- **Data Cleaning**: Processes the raw data by removing duplicates and handling missing values to ensure the quality and reliability of the model.
- **Feature Engineering**: Transforms categorical text data into numerical indexes and combines various data fields into a single features vector necessary for model training.
- **Data Splitting**: Divides the data into training and testing datasets.
- **Model Training and Evaluation**: Multiple machine learning models are trained and evaluated on their ability to predict flight delays. Metrics such as Accuracy, Precision, Recall, F1-Score, and AUC are computed to assess model performance.
- **Visualization**: Uses plots to visually represent the performance of the models, including ROC curves and a bar chart for the various evaluation metrics.

## Running the Project

To execute the project, navigate to the project directory in your terminal and run:
```bash
python <script_name>.py
```
Replace `<script_name>` with the name of the Python script file.

## Expected Results

After running the script, you will see:
- Log messages in the console that provide a step-by-step account of the operations being performed, from data loading through to model evaluation.
- Plots that display the ROC curves and performance metrics for each model, helping you visualize which model performs best in predicting flight delays.

## Conclusion

This project showcases the power of Apache Spark in handling and analyzing large datasets with the goal of predicting flight delays. Through this practical application, one can see how data science can significantly impact decision-making processes in real-world scenarios like air travel.