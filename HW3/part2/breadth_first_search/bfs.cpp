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

void bottom_up_step(Graph graph, solution *sol, int current_distance) {
    int num_nodes = graph->num_nodes;
    
    for (int v = 0; v < num_nodes; v++) {
        // If this node is already visited, skip it
        if (sol->distances[v] != -1) {
            continue;
        }

        // Check if any neighbor has the current distance
        const Vertex* start = incoming_begin(graph, v);
        const Vertex* end = incoming_end(graph, v);
        
        for (const Vertex* neighbor = start; neighbor != end; neighbor++) {
            if (sol->distances[*neighbor] == current_distance) {
                sol->distances[v] = current_distance + 1;
                break; // Move to the next unvisited node after finding a match
            }
        }
    }
}
void bfs_bottom_up(Graph graph, solution *sol) {
    int num_nodes = graph->num_nodes;
    
    // Initialize distances to -1 (indicating unvisited nodes)
    for (int i = 0; i < num_nodes; i++) {
        sol->distances[i] = -1;
    }

    // Assume the BFS starts from node 0
    sol->distances[0] = 0;
    int current_distance = 0;

    // Perform BFS layer by layer
    while (1) {
        int previous_count = 0;
        for (int i = 0; i < num_nodes; i++) {
            if (sol->distances[i] == -1) previous_count++;
        }

        bottom_up_step(graph, sol, current_distance);

        int remaining_count = 0;
        for (int i = 0; i < num_nodes; i++) {
            if (sol->distances[i] == -1) remaining_count++;
        }

        // Stop if no new nodes were visited in this step
        if (previous_count == remaining_count) {
            break;
        }

        current_distance++;
    }
}
void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
