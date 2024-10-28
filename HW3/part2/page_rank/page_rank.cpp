#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;

    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i)
    {
        solution[i] = equal_prob;
    }

    double *scoreOld = (double *) malloc(numNodes * sizeof(double));
    // double *scoreNew = (double *) malloc(numNodes * sizeof(double));

    bool converge = false;

    while (!converge) 
    {
        #pragma omp parallel for
        for (int i = 0; i < numNodes; i++) {
            scoreOld[i] = solution[i];
        }

        #pragma omp parallel for
        for (int vi = 0; vi < numNodes; vi++) {
            double incomingScore = 0.0;

            for (const Vertex *in = incoming_begin(g, vi); in != incoming_end(g, vi); in++) {
                Vertex vj = *in;
                int totalOut = outgoing_size(g, vj);
                if (totalOut > 0) {
                    incomingScore += scoreOld[vj] / totalOut;
                }
            }
            
            solution[vi] = (damping * incomingScore) + ((1.0 - damping) / numNodes);
        }


        double deadSum = 0.0
        #pragma omp parallel for reduciton(+:deadSum)
        for (int v = 0; v < numNodes; v++) {
            if (outgoing_size(g, v) == 0) {
                deadSum += damping * scoreOld[vi];
            }
        }

        deadSum = damping * deadSum / numNodes
        #pragma omp parallel for
        for (int vi = 0; vi < numNodes; vi++) 
            solution[vi] += deadSum;
        

        double globDiff = 0.0;
        #pragma omp parallel for reduciton(+:globDiff)
        for (int vi = 0; vi < numNodes; vi++) {
            globDiff += fabs(solution[vi] - scoreOld[vi]);  
        }
        
        converge = globDiff < convergence;
    }

    free(scoreOld);
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
