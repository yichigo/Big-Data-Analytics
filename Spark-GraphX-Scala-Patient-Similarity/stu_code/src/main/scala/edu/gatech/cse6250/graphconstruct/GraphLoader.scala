/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.graphconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphLoader {
  /**
   * Generate Bipartite Graph using RDDs
   *
   * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
   * @return: Constructed Graph
   *
   */
  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
    medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {

    /** HINT: See Example of Making Patient Vertices Below */
    val vertexPatient: RDD[(VertexId, VertexProperty)] = patients
      .map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))
    
    val numPatient = patients.map(x => x.patientID).distinct().count()

    // Diag
    val diagLast = diagnostics.map(x => ((x.patientID, x.icd9code),x)).reduceByKey((x1,x2) => if(x1.date > x2.date) x1 else x2).map{ case(key,d) => d }
    // Lab
    val labLast = labResults.map(x => ((x.patientID, x.labName),x)).reduceByKey((x1,x2)=> if(x1.date > x2.date) x1 else x2).map{ case(key,d) => d }
    // Med
    val medLast = medications.map(x => ((x.patientID, x.medicine), x)).reduceByKey((x1,x2) => if(x1.date > x2.date) x1 else x2).map{ case(key,v) => v }

    // Diag
    val diagIndex = diagLast.map(_.icd9code).distinct().zipWithIndex().map{ case(v,idx) => (v, idx + numPatient + 1) }
    val numDiag = diagIndex.count()
    val diagVertexId = diagIndex.collect.toMap
    val vertexDiag: RDD[(VertexId, VertexProperty)] = diagIndex.map{ case(code,index)=> (index,DiagnosticProperty(code)) }
    
    // Lab
    val labIndex = labLast.map(_.labName).distinct().zipWithIndex().map{ case(v,idx) => (v, idx + numPatient + numDiag + 1 ) }
    val numLab = labIndex.count()
    val labVertexId = labIndex.collect.toMap
    val vertexLab: RDD[(VertexId, VertexProperty)] = labIndex.map{ case(labname,index) => (index, LabResultProperty(labname)) }

    // Med
    val medIndex = medLast.map(_.medicine).distinct().zipWithIndex().map{ case(v,idx) => (v, idx + numPatient + numDiag + numLab + 1) }
    val medVertexId = medIndex.collect.toMap
    val vertexMed: RDD[(VertexId, VertexProperty)] = medIndex.map{ case(medicine, index) => (index, MedicationProperty(medicine)) }
    
    /**
     * HINT: See Example of Making PatientPatient Edges Below
     *
     * This is just sample edges to give you an example.
     * You can remove this PatientPatient edges and make edges you really need
     */

    // case class PatientPatientEdgeProperty(someProperty: SampleEdgeProperty) extends EdgeProperty
    // val edgePatientPatient: RDD[Edge[EdgeProperty]] = patients
    //   .map({ p =>
    //     Edge(p.patientID.toLong, p.patientID.toLong, SampleEdgeProperty("sample").asInstanceOf[EdgeProperty])
    //   })

    val sc = patients.sparkContext

    val edgePatientDiag = diagLast.map(x => Edge(x.patientID.toLong, diagVertexId(x.icd9code), PatientDiagnosticEdgeProperty(x).asInstanceOf[EdgeProperty]))
    val edgeDiagPatient = diagLast.map(x => Edge(diagVertexId(x.icd9code), x.patientID.toLong, PatientDiagnosticEdgeProperty(x).asInstanceOf[EdgeProperty]))

    val edgePatientLab = labLast.map(x => Edge(x.patientID.toLong, labVertexId(x.labName), PatientLabEdgeProperty(x).asInstanceOf[EdgeProperty]))
    val edgeLabPatient = labLast.map(x => Edge(labVertexId(x.labName), x.patientID.toLong, PatientLabEdgeProperty(x).asInstanceOf[EdgeProperty]))

    val edgePatientMed = medLast.map(x => Edge(x.patientID.toLong, medVertexId(x.medicine), PatientMedicationEdgeProperty(x).asInstanceOf[EdgeProperty]))
    val edgeMedPatient = medLast.map(x => Edge(medVertexId(x.medicine), x.patientID.toLong, PatientMedicationEdgeProperty(x).asInstanceOf[EdgeProperty]))

    // Making Graph
    // val graph: Graph[VertexProperty, EdgeProperty] = Graph[VertexProperty, EdgeProperty](vertexPatient, edgePatientPatient)
    val vertices = sc.union(vertexPatient, vertexDiag, vertexLab, vertexMed)
    val edges = sc.union(edgePatientDiag, edgeDiagPatient, edgePatientLab, edgeLabPatient, edgePatientMed, edgeMedPatient)
    val graph: Graph[VertexProperty, EdgeProperty] = Graph(vertices, edges)

    graph
  }
}
