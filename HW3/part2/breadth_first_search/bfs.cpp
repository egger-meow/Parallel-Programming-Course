#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include<iostream>
using namespace std;
#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
            {
                 int index = __sync_fetch_and_add(&new_frontier->count, 1);
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int current_level
) {
    // Use OpenMP to parallelize the node exploration
    #pragma omp parallel
    {
        // Thread-local new frontier to avoid race conditions
        int local_new_frontier_count = 0;
        int *local_new_frontier = (int*)malloc(g->num_nodes * sizeof(int));

        #pragma omp for
        for (int node = 0; node < g->num_nodes; node++) {
            // Only process unvisited nodes
            if (distances[node] == NOT_VISITED_MARKER) {
                // Locate the range of incoming edges for this node
                int start_edge = g->incoming_starts[node];
                int end_edge = (node == g->num_nodes - 1)
                    ? g->num_edges
                    : g->incoming_starts[node + 1];

                // Check all incoming edges
                for (int edge_idx = start_edge; edge_idx < end_edge; edge_idx++) {
                    int parent = g->incoming_edges[edge_idx];
                    
                    // Check if parent is in current frontier and already visited
                    if (distances[parent] != NOT_VISITED_MARKER) {
                        // Check if parent is actually in the current frontier
                        for (int j = 0; j < frontier->count; j++) {
                            if (frontier->vertices[j] == parent) {
                                // Atomic add to local frontier
                                int local_idx = __sync_fetch_and_add(&local_new_frontier_count, 1);
                                local_new_frontier[local_idx] = node;
                                
                                // Atomic update of distance (to handle potential race)
                                __sync_bool_compare_and_swap(&distances[node], NOT_VISITED_MARKER, current_level);
                                
                                // Break out of inner loops
                                goto next_node;
                            }
                        }
                    }
                }
            }
            next_node:; // Label to break out of nested loops
        }

        // Thread-safe merge of local frontiers
        #pragma omp critical
        {
            for (int i = 0; i < local_new_frontier_count; i++) {
                new_frontier->vertices[new_frontier->count++] = local_new_frontier[i];
            }
        }

        // Free local memory
        free(local_new_frontier);
    }
}

void bfs_bottom_up(Graph graph, solution *sol) {
    int nodes = graph->num_nodes;

    // Larger vertex sets for big graphs
    vertex_set list1, list2;
    vertex_set_init(&list1, nodes);
    vertex_set_init(&list2, nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // Parallel initialization of distances
    #pragma omp parallel for
    for (int i = 0; i < nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // Initialize root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int current_level = 1;

    // Bottom-up BFS traversal
    while (frontier->count > 0) {
        // Clear new frontier before each iteration
        vertex_set_clear(new_frontier);

        // Explore frontier
        bottom_up_step(graph, frontier, new_frontier, sol->distances, current_level);

        // Swap frontiers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        // Increment level
        current_level++;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
