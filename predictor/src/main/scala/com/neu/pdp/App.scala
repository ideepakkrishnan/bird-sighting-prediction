package com.neu.pdp

import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.collection.mutable

/**
 *
 * @author ideepakkrishnan, mohamed_rizwan
 */
object App {

  /**
    * Parses a string and extracts the necessary features into a
    * tuple which can be used for further computations
    * @param line The line of input to be processed
    * @param columnIndexes The columns which are being considered
    *                      for computation
    * @return Extracted features in the form of a Tuple
    */
  def convertToTuple(
         line: String,
         columnIndexes: mutable.HashSet[Int],
         speciesColumn: Int) = {

    val elements: Array[String] = line.split(",")

    // Discard header row and invalid records
    if (elements(0).equals("SAMPLING_EVENT_ID")) {
      null
    } else {

      // Initialize an array to store the features extracted from
      // the current line of input
      val features: Array[Double] = Array.ofDim[Double](columnIndexes.size)

      var currColumnIndex: Int = 0
      var featureIndex: Int = 0

      // Process each element and add the relevant ones into the
      // feature array
      elements.foreach(element => {
        // Check if this column is to be considered
        if (columnIndexes.contains(currColumnIndex)) {
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
      (elements(0), Vectors.dense(features))
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
    * @param record The record
    * @param lrModel The model to be used
    * @return The updated LabeledPoint
    */
  def appendLRPrediction(
          record: (String, org.apache.spark.mllib.linalg.Vector),
          lrModel: LogisticRegressionModel) = {

    // Return the prediction
    (record._1,
      Vectors.dense(
        record._2.toArray :+
          predictUsingLogisticRegression(record._2, lrModel)))
  }

  /**
    * Performs predictions using the specified models and
    * calculates an average prediction using the results
    * from all these models
    * @param record The features
    * @param gbtModel A RandomForestModel
    * @return The average prediction
    */
  def calculatePrediction(
           record: (String, org.apache.spark.mllib.linalg.Vector),
           gbtModel: GradientBoostedTreesModel): (String, Double) = {

    // Return the prediction
    (record._1, predictUsingGradientBoostTrees(record._2, gbtModel))
  }

  /**
    * The main method for this application
    * @param args Runtime arguments
    */
  def main(args: Array[String]) {
    if (args.length != 5) {
      println(
          "Invalid number of arguments. The expected arguments " +
          "are: Logistic Regression model path, gradient boosted " +
          "model path, unlabelled data path, output folder")
    } else {
      // Initialize job configuration
      val conf = new SparkConf().setAppName("Sighting Prediction")

      // Initialize job context
      val sc = new SparkContext(conf)

      // Read the input file path, output folder path and the column
      // index of the species for which we are running the prediction
      val lrModelPath = args(0)
      val gbtModelPath = args(1)
      val inputPath = args(2)
      val outputPath = args(3)
      val speciesColumn = args(4).toInt

      val inputRDD: RDD[String] = sc.textFile(inputPath)

      val lrModel : LogisticRegressionModel =
            LogisticRegressionModel.load(sc, lrModelPath)

      val gbtModel : GradientBoostedTreesModel =
            GradientBoostedTreesModel.load(sc, gbtModelPath)

      // Store the columns being considered for calculating the
      // prediction inside a HashSet so that it can be used in
      // the map phase to extract the required values
      var arrColumns = Array[Int](2,3,5,6,12,13,14,16,
        955,956,957,958,959,960,962)
      arrColumns = arrColumns ++ Array.range(963, 1015)
      arrColumns = arrColumns ++ Array.range(1019, 1089)
      arrColumns = arrColumns ++ Array.range(1090, 1102)

      // Generate a hashset out of the columns to make sure
      // there are no duplicates
      val hsColumns: mutable.HashSet[Int] = new mutable.HashSet[Int]()
      arrColumns.foreach(value => hsColumns.add(value))

      // RDD storing extracted features as LabeledPoint-s
      val extractedData: RDD[(String, org.apache.spark.mllib.linalg.Vector)] = inputRDD
            .map(line => convertToTuple(line, hsColumns, speciesColumn))
            .filter(x => x != null).persist()

      // Perform the prediction for each record using the
      // above model and append it to each record. This
      // step is done as a step to boost the accuracy while
      // running gradient boosted model
      val lrProcessedData: RDD[(String, org.apache.spark.mllib.linalg.Vector)] =
            extractedData.map(
                  record => appendLRPrediction(
                                record,
                                lrModel))

      // Generate the final predictions using gradient
      // boosted trees model
      val gbtResultRDD: RDD[(String, Double)] =
            lrProcessedData.map(
                point => calculatePrediction(
                              point,
                              gbtModel))

      gbtResultRDD.map(line => {line._1 + "," + line._2})
                  .saveAsTextFile(outputPath)
    }
  }

}
