#include "page_rank.h"
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include "../common/CycleTimer.h"
#include "../common/graph.h"

void pageRank(Graph g, double *solution, double damping, double convergence)
{
    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    
    // Initialize solution array
    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    double *scoreOld = (double *) malloc(numNodes * sizeof(double));
    bool converge = false;
    
    while (!converge) {
        // Copy current scores to old scores
        #pragma omp parallel for
        for (int i = 0; i < numNodes; i++) {
            scoreOld[i] = solution[i];
        }

        // First, compute basic PageRank contribution
        #pragma omp parallel for
        for (int vi = 0; vi < numNodes; vi++) {
            double incomingScore = 0.0;
            
            // Sum up contributions from all incoming edges
            for (const Vertex *in = incoming_begin(g, vi); in != incoming_end(g, vi); ++in) {
                Vertex vj = *in;
                int totalOut = outgoing_size(g, vj);
                incomingScore += scoreOld[vj] / totalOut;
            }
            
            // Apply damping factor
            solution[vi] = (damping * incomingScore) + ((1.0 - damping) / numNodes);
        }

        // Handle dangling nodes (nodes with no outgoing edges)
        double danglingSum = 0.0;
        #pragma omp parallel for reduction(+:danglingSum)
        for (int vi = 0; vi < numNodes; vi++) {
            if (outgoing_size(g, vi) == 0) {
                danglingSum += scoreOld[vi];
            }
        }
        
        // Distribute dangling node contributions
        double danglingContrib = (damping * danglingSum) / numNodes;
        #pragma omp parallel for
        for (int vi = 0; vi < numNodes; vi++) {
            solution[vi] += danglingContrib;
        }

        // Check for convergence
        double globDiff = 0.0;
        #pragma omp parallel for reduction(+:globDiff)
        for (int vi = 0; vi < numNodes; vi++) {
            globDiff += fabs(solution[vi] - scoreOld[vi]);
        }
        
        converge = (globDiff < convergence);
    }

    free(scoreOld);
}