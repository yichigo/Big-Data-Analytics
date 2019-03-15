package edu.gatech.cse6250.randomwalk

import edu.gatech.cse6250.model.{ PatientProperty, EdgeProperty, VertexProperty }
import org.apache.spark.graphx._

object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {
    /**
     * Given a patient ID, compute the random walk probability w.r.t. to all other patients.
     * Return a List of patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay
     */

    /** Remove this placeholder and implement your code */
    // List(1, 2, 3, 4, 5)

    val src: VertexId = patientID

    // initialize PageRank Graph
    var rankGraph: Graph[Double, Double] = graph
      .outerJoinVertices( graph.outDegrees)( (vid, vdata, deg) => deg.getOrElse(0) )
      .mapTriplets( e => 1.0 / e.srcAttr, TripletFields.Src )
      .mapVertices( (id, attr) => if (!(id != src)) alpha else 0.0 )

    // Random Walk
    var i = 0
    var prevRankGraph: Graph[Double, Double] = null
    while (i < numIter) {
      rankGraph.cache()

      val rankUpdates = rankGraph.aggregateMessages[Double](
        ctx => ctx.sendToDst(ctx.srcAttr * ctx.attr), _ + _, TripletFields.Src)

      prevRankGraph = rankGraph

      rankGraph = rankGraph.joinVertices(rankUpdates) {
        (id, oldRank, msgSum) => ( if (id==patientID) alpha else 0.0 ) + ( 1.0 - alpha ) * msgSum
      }.cache()

      rankGraph.edges.foreachPartition(x => {})
      prevRankGraph.vertices.unpersist(false)
      prevRankGraph.edges.unpersist(false)

      i += 1
    }

    val patientIDs = graph.vertices.filter(x => x._2.isInstanceOf[PatientProperty]).map(_._1).collect.toSet
    val top10 = rankGraph.vertices.filter(x => patientIDs.contains(x._1)).takeOrdered(11)(Ordering[Double].reverse.on(x => x._2)).map(_._1)
    val result = top10.slice(1, top10.length).toList

    result
  }
}
