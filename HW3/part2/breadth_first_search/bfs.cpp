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
    int i
) {
    // Check the incoming edges for this vertex
    int start_edge = g->incoming_starts[i];
    int end_edge = (i == g->num_nodes - 1) 
        ? g->num_edges 
        : g->incoming_starts[i + 1];

    // Iterate through incoming edges
    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
        int parent = g->incoming_edges[neighbor];

        // Check if the parent is already visited and in the current frontier
        if (distances[parent] != NOT_VISITED_MARKER) {
            // Check if the parent is in the current frontier
            for (int j = 0; j < frontier->count; j++) {
                if (frontier->vertices[j] == parent) {
                    if (__sync_bool_compare_and_swap(&distances[i], NOT_VISITED_MARKER, distances[parent] + 1)) {
                        // Atomically add vertex to new frontier
                        int idx = __sync_fetch_and_add(&new_frontier->count, 1);
                        new_frontier->vertices[idx] = i;
                    }
                    return;
                }
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    int nodes = graph->num_nodes;

    // Initialize two vertex sets for frontiers
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, nodes);
    vertex_set_init(&list2, nodes);

    // Set up frontier pointers
    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // Initialize all distances to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Start BFS from root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // Continue until no more nodes can be reached
    do {
        // Clear the new frontier
        vertex_set_clear(new_frontier);

        // Explore unvisited nodes
        #pragma omp parallel for
        for (int i = 0; i < nodes; i++) {
            // Only process unvisited nodes
            if (sol->distances[i] == NOT_VISITED_MARKER) {
                bottom_up_step(graph, frontier, new_frontier, sol->distances, i);
            }
        }

        // Swap frontiers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

    } while (frontier->count != 0);

    // Note: vertex sets will be automatically freed by the caller
}
void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
