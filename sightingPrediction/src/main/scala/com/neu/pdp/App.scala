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
        elements(speciesColumn).equals("?") ||
        elements(speciesColumn).equals("X")) { // Check invalid records
      null
    } else {

      // Initialize an array to store the features extracted from
      // the current line of input
      val features: Array[Double] = Array.ofDim[Double](columnIndexes.size)

      // Initialize the label for this record
      if (elements(speciesColumn).toInt > 0) {
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
    * @param gbtModel A RandomForestModel
    * @return The average prediction
    */
  def calculatePrediction(
          labeledPoint: LabeledPoint,
          gbtModel: GradientBoostedTreesModel): (Double, Double) = {

    // Return the prediction
    (labeledPoint.label, predictUsingGradientBoostTrees(labeledPoint, gbtModel))
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
            955,960)
      arrColumns = arrColumns ++ Array.range(963, 1015)
      arrColumns = arrColumns ++ Array.range(1019, 1089)
      arrColumns = arrColumns ++ Array.range(1090, 1102)
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
      categoricalFeaturesInfo += (2 -> 13) // Months : 1 - 12
      categoricalFeaturesInfo += (3 -> 367) // Days : 1-366
      categoricalFeaturesInfo += (9 -> 38) // BCR : 1-38
      categoricalFeaturesInfo += (10 -> 121) // OMERNIK_L3_ECOREGION : 1-121

      // Prepare random samples for all models
      /*val randomTrainingData: RDD[(Int, LabeledPoint)] =
            trainingData.map(line => (scala.util.Random.nextInt(4), line))*/

      // Define parameters for DecisionTree
      val numClasses = 2
      /*val impurity = "gini"
      val maxDecisionTreeDepth = 9
      val maxBins = 4000*/

      /*val dtModel: DecisionTreeModel =
            DecisionTree.trainClassifier(
              trainingData,  // Training data
              numClasses, categoricalFeaturesInfo,
              impurity, maxDecisionTreeDepth, maxBins)

      // Define parameters for RandomForest
      val algoStrategy = "Classification"
      val numTrees = 100
      val subsetStrategy = "auto"  // Feature subset strategy -> Let the algorithm choose

      val rfModel: RandomForestModel =
            RandomForest.trainClassifier(
              trainingData,  // Training data
              Strategy.defaultStrategy(algoStrategy),
              numTrees, subsetStrategy, maxBins)*/

      // Define parameters for GradientBoostedTrees
      val activeStrategy = "Classification"
      val numBoostingIterations = 50
      val maxBoostingDepth = 9

      var boostingStrategy = BoostingStrategy.defaultParams(activeStrategy)
      boostingStrategy.setNumIterations(numBoostingIterations)
      boostingStrategy.treeStrategy.setNumClasses(numClasses)
      boostingStrategy.treeStrategy.setMaxDepth(maxBoostingDepth)

      val gbtModel: GradientBoostedTreesModel =
            GradientBoostedTrees.train(
              trainingData,  // Training data
              boostingStrategy)

      /*val lrModel: LogisticRegressionModel =
            new LogisticRegressionWithLBFGS()
                  .setNumClasses(2)
                  .run(trainingData)*/

      // This is the validation step to calculate the accuracy of the validation data
      val ensembleResultRDD: RDD[(Double, Double)] =
            testData.map(
              point => calculatePrediction(
                          point,
                          gbtModel))

      val finalAccuracy = ensembleResultRDD
                              .filter(r => r._1 == r._2)
                              .count
                              .toDouble / testData.count()

      println("Accuracy = " + finalAccuracy)

      //dtModel.save(sc, outputPath + "/DecisionTreeModel")
      //rfModel.save(sc, outputPath + "/RandomForestModel")
      gbtModel.save(sc, outputPath + "/GradientBoostModel")
      //lrModel.save(sc, outputPath + "/LogisticRegressionModel")

      sc.stop()

    } else {
      println("Invalid run time arguments")
    }
  }

}
