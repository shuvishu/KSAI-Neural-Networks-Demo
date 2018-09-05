package edu.knoldus.ksai.demo

import breeze.linalg.DenseMatrix
import ksai.core.classification.{LeastMeanSquares, LogisticSigmoid, Network}
import ksai.data.parser.DelimitedParser

object NeuralNetworkDemo extends App {
  val trainFile: String = getClass.getResource("/zip_less.train").getPath
  val parsedResultTrain = DelimitedParser.parse(trainFile)
  val testFile: String = getClass.getResource("/zip_less_test.train").getPath
  val parsedResultTest = DelimitedParser.parse(testFile)
  val network = Network(LeastMeanSquares, LogisticSigmoid,256, parsedResultTrain.getNumericTargets.max + 1)
  val trained = network.learn(DenseMatrix(parsedResultTrain.data: _*), parsedResultTrain.getNumericTargets.toArray)
  val predictedResult =  (parsedResultTest.data zip parsedResultTest.getNumericTargets).map {
    case (arr, actualOutput) => trained.predict(arr) == actualOutput
  }

  val (nonErrors, errors) = predictedResult.partition(isError => isError)

  println(s"Non Errors [${nonErrors.length}] Errors [${errors.length}]")

}
