/**
 *
 * students: please put your implementation in this file!
 */
package edu.gatech.cse6250.jaccard

import edu.gatech.cse6250.model._
import edu.gatech.cse6250.model.{ EdgeProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /**
     * Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients.
     * Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay. The given patientID should be excluded from the result.
     */

    /** Remove this placeholder and implement your code */
    // List(1, 2, 3, 4, 5)
    val patientIds = graph.vertices.filter(x => x._2.isInstanceOf[PatientProperty]).map(_._1).collect.toSet
    val patientNeighbors = graph.collectNeighborIds(EdgeDirection.Out).filter(x => patientIds.contains(x._1))
    val patientIdNeighbor = patientNeighbors.filter(_._1 == patientID).map(_._2).flatMap(x => x).collect().toSet
    val jaccardScores = patientNeighbors.filter(_._1 != patientID).map(x => (x._1, jaccard(patientIdNeighbor, x._2.toSet)))
    val result = jaccardScores.takeOrdered(10)(Ordering[Double].reverse.on(_._2)).map(_._1).toList

    result
  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    /**
     * Given a patient, med, diag, lab graph, calculate pairwise similarity between all
     * patients. Return a RDD of (patient-1-id, patient-2-id, similarity) where
     * patient-1-id < patient-2-id to avoid duplications
     */

    /** Remove this placeholder and implement your code */
    // val sc = graph.edges.sparkContext
    // sc.parallelize(Seq((1L, 2L, 0.5d), (1L, 3L, 0.4d)))
    val patientIds = graph.vertices.filter(x => x._2.isInstanceOf[PatientProperty]).map(_._1).collect.toSet
    val patientNeighbors = graph.collectNeighborIds(EdgeDirection.Out).filter(x => patientIds.contains(x._1))
    val patientCartesianNeighbors = patientNeighbors.cartesian(patientNeighbors).filter(x => x._1._1 < x._2._1)
    val result = patientCartesianNeighbors.map(x => (x._1._1, x._2._1, jaccard(x._1._2.toSet, x._2._2.toSet)))

    result
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /**
     * Helper function
     *
     * Given two sets, compute its Jaccard similarity and return its result.
     * If the union part is zero, then return 0.
     */

    /** Remove this placeholder and implement your code */
    // 0.0
    if (a.isEmpty || b.isEmpty) {
        return 0.0
    }
    val result = a.intersect(b).size.toDouble / a.union(b).size.toDouble
    result
  }
}
