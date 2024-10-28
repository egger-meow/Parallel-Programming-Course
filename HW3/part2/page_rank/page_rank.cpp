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
    for (int i = 0; i < numNodes; ++i)
    {
        solution[i] = equal_prob;
    }
 double *score_old = (double *)malloc(numNodes * sizeof(double));
    double *score_new = (double *)malloc(numNodes * sizeof(double));

    bool converged = false;

    while (!converged) {
        // Copy current scores to score_old
        #pragma omp parallel for
        for (int i = 0; i < numNodes; ++i) {
            score_old[i] = solution[i];
            score_new[i] = 0.0; // Reset score_new
        }

        // Compute new scores
        #pragma omp parallel for
        for (int vi = 0; vi < numNodes; ++vi) {
            double incoming_score = 0.0;

            // Sum scores from incoming edges
            for (const Vertex *in = incoming_begin(g, vi); in != incoming_end(g, vi); ++in) {
                Vertex vj = *in;
                int out_degree = outgoing_size(g, vj);
                if (out_degree > 0) {
                    incoming_score += score_old[vj] / out_degree;
                }
            }

            // Apply damping factor
            score_new[vi] = (damping * incoming_score) + ((1.0 - damping) / numNodes);
        }

        // Handle nodes with no outgoing edges
        #pragma omp parallel for
        for (int vi = 0; vi < numNodes; ++vi) {
            if (outgoing_size(g, vi) == 0) {
                score_new[vi] += (damping * score_old[vi]) / numNodes;
            }
        }

        // Calculate convergence
        double global_diff = 0.0;
        #pragma omp parallel for reduction(+:global_diff)
        for (int i = 0; i < numNodes; ++i) {
            global_diff += fabs(score_new[i] - score_old[i]);
        }

        // Check for convergence
        converged = (global_diff < convergence);

        // Update solution with new scores
        #pragma omp parallel for
        for (int i = 0; i < numNodes; ++i) {
            solution[i] = score_new[i];
        }
    }

    // Clean up
    free(score_old);
    free(score_new);
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
