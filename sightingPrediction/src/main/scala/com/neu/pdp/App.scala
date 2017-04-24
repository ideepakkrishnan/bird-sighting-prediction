package com.neu.pdp

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.collection.mutable

/**
  * @author ideepakkrishnan, mohamed_rizwan
  */
object App {

  /**
    * Parses a string and extracts the necessary features into a
    * labelled point which can be used for further computations
    * @param line The line of input to be processed
    * @param columnIndexes The columns which are being considered
    *                      for computation
    * @return Extracted features in the form of a LabeledPoint
    */
  def convertToLabeledPoint(
         line: String,
         columnIndexes: mutable.HashSet[Int],
         speciesColumn: Int) : LabeledPoint = {

    val elements: Array[String] = line.split(",")

    // Discard header row and invalid records
    if (elements(0).equals("SAMPLING_EVENT_ID") || // Check for title row
        elements(speciesColumn).equals("?") ||
        elements(speciesColumn).equals("X")) { // Check invalid records
      null
    } else {

      // Initialize an array to store the features extracted from
      // the current line of input
      val features: Array[Double] = Array.ofDim[Double](columnIndexes.size)

      // Initialize the label for this record
      if (elements(speciesColumn).toInt > 0) {
        features(0) = 1
      } else {
        features(0) = 0
      }

      var currColumnIndex: Int = 0
      var featureIndex: Int = 1

      // Process each element and add the relevant ones into the
      // feature array
      elements.foreach(element => {
        // Check if this column is to be considered
        if (columnIndexes.contains(currColumnIndex) &&
            currColumnIndex != speciesColumn) {
          // Check if the column has a valid value
          if (element.trim.equals("?") ||
              element.trim.equals("X")) {
            // Means that the value is invalid. Replace it with 0.
            features(featureIndex) = 0.0
          }
          else {
            // Update the value inside corresponding index of
            // feature array
            features(featureIndex) = element.toDouble
          }
          featureIndex += 1
        }
        currColumnIndex += 1
      })

      // Generate the LabeledPoint using extracted features
      LabeledPoint(features(0), Vectors.dense(features.tail))
    }
  }

  /**
    * Predict the outcome using the specified
    * GradientBoostedTreesModel
    * @param features The features
    * @param gbtModel The specified model
    * @return The prediction
    */
  def predictUsingGradientBoostTrees(
         features: org.apache.spark.mllib.linalg.Vector,
         gbtModel: GradientBoostedTreesModel) : Double = {
    gbtModel.predict(features)
  }

  /**
    * Predict the outcome using the specified
    * LogisticRegressionModel
    * @param features The features
    * @param lrModel The specified model
    * @return The prediction
    */
  def predictUsingLogisticRegression(
          features: org.apache.spark.mllib.linalg.Vector,
          lrModel: LogisticRegressionModel) : Double = {
    lrModel.predict(features)
  }

  /**
    * Predicts the outcome for the record passed
    * in as argument and appends it to the features
    * of it's LabeledPoint
    * @param labeledPoint The record
    * @param lrModel The model to be used
    * @return The updated LabeledPoint
    */
  def appendLRPrediction(
         labeledPoint: LabeledPoint,
         lrModel: LogisticRegressionModel): LabeledPoint = {

    // Return the prediction
    LabeledPoint(
      labeledPoint.label,
      Vectors.dense(
        labeledPoint.features.toArray :+
          predictUsingLogisticRegression(
            labeledPoint.features, lrModel)))
  }

  /**
    * Performs predictions using the specified models and
    * calculates an average prediction using the results
    * from all these models
    * @param labeledPoint The features
    * @param gbtModel A RandomForestModel
    * @return The average prediction
    */
  def calculatePrediction(
          labeledPoint: LabeledPoint,
          gbtModel: GradientBoostedTreesModel): (Double, Double) = {

    // Return the prediction
    (labeledPoint.label,
      predictUsingGradientBoostTrees(
        labeledPoint.features, gbtModel))
  }

  /**
    * The main method for this application
    * @param args Runtime arguments
    */
  def main(args: Array[String]) {
    if (args.length == 3) {
      // Initialize job configuration
      val conf = new SparkConf().setAppName("Sighting Prediction")

      // Initialize job context
      val sc = new SparkContext(conf)

      // Read the input file path, output folder path and the column
      // index of the species for which we are running the prediction
      val inputPath = args(0)
      val outputPath = args(1)
      val speciesColumn = args(2).toInt

      val inputRDD: RDD[String] = sc.textFile(inputPath)

      // Store the columns being considered for calculating the
      // prediction inside a HashSet so that it can be used in
      // the map phase to extract the required values
      var arrColumns = Array[Int](2,3,5,6,12,13,14,16,speciesColumn,
            955,956,957,958,959,960,962)
      arrColumns = arrColumns ++ Array.range(963, 1015)
      arrColumns = arrColumns ++ Array.range(1019, 1089)
      arrColumns = arrColumns ++ Array.range(1090, 1102)

      // Generate a hashset out of the columns to make sure
      // there are no duplicates
      val hsColumns: mutable.HashSet[Int] = new mutable.HashSet[Int]()
      arrColumns.foreach(value => hsColumns.add(value))

      // RDD storing extracted features as LabeledPoint-s
      val extractedData: RDD[LabeledPoint] = inputRDD
            .map(line => convertToLabeledPoint(line, hsColumns, speciesColumn))
            .filter(x => x != null).persist()

      // Initialize categorical fields for DecisionTree.
      // This specifies the number of unique values that
      // each column can take. Though optional, this is
      // documented to churn out better performance.
      var categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]()
      categoricalFeaturesInfo += (2 -> 13) // Months : 1 - 12
      categoricalFeaturesInfo += (3 -> 367) // Days : 1-366
      categoricalFeaturesInfo += (14 -> 38) // BCR : 1-38
      categoricalFeaturesInfo += (15 -> 121) // OMERNIK_L3_ECOREGION : 1-121

      // Define parameters for DecisionTree
      val numClasses = 2

      // Initialize the Logistic Regression model
      val lrModel: LogisticRegressionModel =
            new LogisticRegressionWithLBFGS()
                  .setNumClasses(2)
                  .run(extractedData)

      // Perform the prediction for each record using the
      // above model and append it to each record. This
      // step is done as a step to boost the accuracy while
      // running gradient boosted model
      val lrProcessedData: RDD[LabeledPoint] =
            extractedData.map(
              record => appendLRPrediction(
                            record,
                            lrModel))

      // Split the RDD into training and validation data
      val splits: Array[RDD[LabeledPoint]] =
            lrProcessedData.randomSplit(Array(0.8, 0.2))
      val trainingData: RDD[LabeledPoint] = splits(0)
      val testData: RDD[LabeledPoint] = splits(1)

      // Define parameters for GradientBoostedTrees
      val activeStrategy = "Classification"
      val numBoostingIterations = 120
      val maxBoostingDepth = 12

      // Initialize the boosting strategy for gradient
      // boosted trees model
      val boostingStrategy = BoostingStrategy.defaultParams(activeStrategy)
      boostingStrategy.setNumIterations(numBoostingIterations)
      boostingStrategy.treeStrategy.setNumClasses(numClasses)
      boostingStrategy.treeStrategy.setMaxDepth(maxBoostingDepth)

      // Initialize the gradient boosted trees model
      val gbtModel: GradientBoostedTreesModel =
            GradientBoostedTrees.train(
              trainingData,  // Training data
              boostingStrategy)

      // Generate the final predictions using gradient
      // boosted trees model
      val gbtResultRDD: RDD[(Double, Double)] =
            testData.map(
              point => calculatePrediction(
                          point,
                          gbtModel))

      // Calculate the accuracy of our predictions
      // from gradient boosted trees model and write
      // it to sysout
      val finalAccuracy = gbtResultRDD
                              .filter(r => r._1 == r._2)
                              .count
                              .toDouble / testData.count()

      println("Accuracy = " + finalAccuracy)

      // Save the models for later use
      lrModel.save(sc, outputPath + "/LogisticRegressionModel")
      gbtModel.save(sc, outputPath + "/GradientBoostModel")

      sc.stop()

    } else {
      println(
        "Invalid run time arguments. Please specify the " +
        "following arguments: input folder, output folder, " +
        "species column.")
    }
  }

}
