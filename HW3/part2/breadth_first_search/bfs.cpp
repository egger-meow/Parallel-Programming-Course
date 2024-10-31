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
    int i,
    int *local_new_frontier_count,
    int *local_new_frontier
) {
    int start_edge = g->incoming_starts[i];
    int end_edge = (i == g->num_nodes - 1)
                ? g->num_edges
                : g->incoming_starts[i + 1];
    
    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
        int parent = g->incoming_edges[neighbor];
        
        // Check if parent is already visited
        if (distances[parent] != NOT_VISITED_MARKER) {
            // Check if parent is in the current frontier
            for (int j = 0; j < frontier->count; j++) {
                if (frontier->vertices[j] == parent) {
                    // Use atomic operation or local buffer to avoid race conditions
                    int idx = __sync_fetch_and_add(local_new_frontier_count, 1);
                    local_new_frontier[idx] = i;
                    distances[i] = distances[parent] + 1;
                    return;
                }
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    int nodes = graph->num_nodes;

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, nodes);
    vertex_set_init(&list2, nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // Initialize distances
    #pragma omp parallel for
    for (int i = 0; i < nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    do {
        // Local buffers for thread-safe frontier expansion
        int local_new_frontier_count = 0;
        int *local_new_frontier = (int*)malloc(nodes * sizeof(int));
        
        #pragma omp parallel
        {
            // Local thread-private variables to reduce contention
            int thread_local_count = 0;
            int *thread_local_frontier = (int*)malloc(nodes * sizeof(int));
            
            #pragma omp for
            for (int i = 0; i < nodes; i++) {
                if (sol->distances[i] == NOT_VISITED_MARKER) {
                    bottom_up_step(graph, frontier, new_frontier, sol->distances, 
                                   i, &local_new_frontier_count, local_new_frontier);
                }
            }
            
            free(thread_local_frontier);
        }
        
        // Copy local new frontier to actual new frontier
        new_frontier->count = local_new_frontier_count;
        memcpy(new_frontier->vertices, local_new_frontier, 
               local_new_frontier_count * sizeof(int));
        
        free(local_new_frontier);
        
        // Swap frontiers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        
        // Clear the new frontier for next iteration
        vertex_set_clear(new_frontier);
    } while(frontier->count != 0);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
