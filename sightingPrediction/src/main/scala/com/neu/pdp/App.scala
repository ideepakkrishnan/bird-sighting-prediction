package com.neu.pdp

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, GradientBoostedTreesModel, RandomForestModel}
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Strategy}
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
         speciesColumn: Int) = {

    val elements: Array[String] = line.split(",")

    // Discard header row and invalid records
    if (elements(0).equals("SAMPLING_EVENT_ID") || // Check for title row
        elements(speciesColumn).equals("?") || // Check invalid records
        elements(speciesColumn).equals("X")) {
      null
    } else {

      // Initialize an array to store the features extracted from
      // the current line of input
      val features: Array[Double] = Array.ofDim[Double](columnIndexes.size)

      // Initialize the label for this record
      if (elements(speciesColumn).toInt >= 0) {
        features(0) = 1;
      } else {
        features(0) = 0;
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
    * DecisionTreeModel
    * @param labeledPoint The features
    * @param dtModel The specified model
    * @return The prediction
    */
  def predictUsingDecisionTree(
          labeledPoint: LabeledPoint,
          dtModel: DecisionTreeModel) : Double = {
    dtModel.predict(labeledPoint.features)
  }

  /**
    * Predict the outcome using the specified
    * GradientBoostedTreesModel
    * @param labeledPoint The features
    * @param gbtModel The specified model
    * @return The prediction
    */
  def predictUsingGradientBoostTrees(
         labeledPoint: LabeledPoint,
         gbtModel: GradientBoostedTreesModel) : Double = {
    gbtModel.predict(labeledPoint.features)
  }

  /**
    * Predict the outcome using the specified
    * RandomForestModel
    * @param labeledPoint The features
    * @param rfModel The specified model
    * @return The prediction
    */
  def predictUsingRandomForest(
          labeledPoint: LabeledPoint,
          rfModel: RandomForestModel) : Double = {
    rfModel.predict(labeledPoint.features)
  }

  /**
    * Predict the outcome using the specified
    * LogisticRegressionModel
    * @param labeledPoint The features
    * @param lrModel The specified model
    * @return The prediction
    */
  def predictUsingLogisticRegression(
          labeledPoint: LabeledPoint,
          lrModel: LogisticRegressionModel) : Double = {
    lrModel.predict(labeledPoint.features)
  }

  /**
    * Calculates the average for a list of double
    * values
    * @param values A list of Double-s
    * @return Calculated average
    */
  def findAverage(values: List[Double]) : Double = {
    var sum : Double = 0.0

    if (values.nonEmpty) {
      values.foreach(value => { sum += value })
      return sum / values.size
    }

    sum
  }

  /**
    * Performs predictions using the specified models and
    * calculates an average prediction using the results
    * from all these models
    * @param labeledPoint The features
    * @param dtModel A DecisionTreeModel
    * @param gbtModel A GradientBoostedTreesModel
    * @param rfModel A RandomForestModel
    * @param lrModel A LogisticRegressionModel
    * @return The average prediction
    */
  def calculatePrediction(
          labeledPoint: LabeledPoint,
          dtModel: DecisionTreeModel,
          gbtModel: GradientBoostedTreesModel,
          rfModel: RandomForestModel,
          lrModel: LogisticRegressionModel): (Double, Double) = {

    // Calculate the predictions from each model and store
    // them as a list
    val predictions: List[Double] =
          List(
            predictUsingDecisionTree(labeledPoint, dtModel),
            predictUsingGradientBoostTrees(labeledPoint, gbtModel),
            predictUsingRandomForest(labeledPoint, rfModel),
            predictUsingLogisticRegression(labeledPoint, lrModel))

    // Calculate the average prediction
    var avgPrediction: Double = findAverage(predictions)

    if (avgPrediction < 0.50)
      avgPrediction = 0
    else
      avgPrediction = 1

    // Return the prediction
    (labeledPoint.label, avgPrediction)
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
      val arrColumns = Array[Int](2,3,5,6,12,13,14,16,speciesColumn,
            955,960,962,963,964,965,966,967)
      val hsColumns: mutable.HashSet[Int] = new mutable.HashSet[Int]()
      arrColumns.foreach(value => hsColumns.add(value))

      // RDD storing extracted features as LabeledPoint-s
      val extractedData: RDD[LabeledPoint] = inputRDD
            .map(line => convertToLabeledPoint(line, hsColumns, speciesColumn))
            .filter(x => x != null).persist()

      // Split the RDD into training and validation data
      val splits: Array[RDD[LabeledPoint]] = extractedData.randomSplit(Array(0.8, 0.2))
      val trainingData: RDD[LabeledPoint] = splits(0)
      val testData: RDD[LabeledPoint] = splits(1)

      // Initialize categorical fields for DecisionTree.
      // This specifies the number of unique values that
      // each column can take. Though optional, this is
      // documented to churn out better performance.
      var categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]()
      categoricalFeaturesInfo += (2 -> 31)  // Days 1-31
      categoricalFeaturesInfo += (3 -> 366)
      categoricalFeaturesInfo += (9 -> 38)  // BCR 1-38
      categoricalFeaturesInfo += (10 -> 121)  // OMERNIK_L3_ECOREGION 1-121

      // Prepare random samples for all models
      val randomTrainingData: RDD[(Int, LabeledPoint)] =
            trainingData.map(line => (scala.util.Random.nextInt(4), line))

      val dtModel: DecisionTreeModel =
            DecisionTree.trainClassifier(
              randomTrainingData
                .filter(x => x._1 != 0 || x._1 == null)
                .map(x => x._2),  // Training data
              2,  // Number of classes
              categoricalFeaturesInfo,
              "gini",  // Impurity
              9,  // Max Depth
              4000)  // Max bins

      val rfModel: RandomForestModel =
            RandomForest.trainClassifier(
              randomTrainingData
                .filter(x => x._1 != 1 || x._1 == null)
                .map(x => x._2),  // Training data
              Strategy.defaultStrategy("Classification"),
              10,  // Number of trees
              "auto",  // Feature subset strategy -> Let the algorithm choose
              4000)  // Max bins

      var boostingStrategy = BoostingStrategy.defaultParams("Classification")
      boostingStrategy.setNumIterations(5)
      boostingStrategy.treeStrategy.setNumClasses(2)
      boostingStrategy.treeStrategy.setMaxDepth(9)

      val gbtModel: GradientBoostedTreesModel =
            GradientBoostedTrees.train(
              randomTrainingData
                .filter(x => x._1 != 2 || x._1 == null)
                .map(x => x._2),  // Training data
              boostingStrategy)

      val lrModel: LogisticRegressionModel =
            new LogisticRegressionWithLBFGS()
                  .setNumClasses(2)
                  .run(
                    randomTrainingData
                      .filter(x => x._1 != 3 || x._1 == null)
                      .map(x => x._2))

      // This is the validation step to calculate the accuracy of the validation data
      val ensembleResultRDD: RDD[(Double, Double)] =
            testData.map(
              point => calculatePrediction(
                          point,
                          dtModel,
                          gbtModel,
                          rfModel,
                          lrModel))

      val finalAccuracy = ensembleResultRDD
                              .filter(r => r._1 == r._2)
                              .count
                              .toDouble / testData.count()

      println("Accuracy = " + finalAccuracy)

      dtModel.save(sc, args(1) + "/DecisionTreeModel")
      rfModel.save(sc, args(1) + "/RandomForestModel")
      gbtModel.save(sc, args(1) + "/GradientBoostModel")
      lrModel.save(sc, args(1) + "/LogisticRegressionModel")

      sc.stop()

    } else {
      println("Invalid run time arguments")
    }
  }

}
