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

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    int i,
    std::vector<int>& local_frontier)
{
    int start_edge = g->incoming_starts[i];
    int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];

    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
        int parent = g->incoming_edges[neighbor];

        if (distances[parent] != NOT_VISITED_MARKER) {
            if (distances[i] == NOT_VISITED_MARKER) {
                distances[i] = distances[parent] + 1;
                local_frontier.push_back(i);
                return;
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    int nodes = graph->num_nodes;

    vertex_set list1, list2;
    vertex_set_init(&list1, nodes);
    vertex_set_init(&list2, nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    #pragma omp parallel for
    for (int i = 0; i < nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    do {
        vertex_set_clear(new_frontier);

        // Use thread-local frontiers to reduce contention
        #pragma omp parallel
        {
            std::vector<int> local_frontier;

            #pragma omp for nowait
            for (int i = 0; i < nodes; i++) {
                if (sol->distances[i] == NOT_VISITED_MARKER) {
                    bottom_up_step(graph, frontier, sol->distances, i, local_frontier);
                }
            }

            // Merge local frontiers into the global new_frontier
            #pragma omp critical
            {
                for (int v : local_frontier) {
                    new_frontier->vertices[new_frontier->count++] = v;
                }
            }
        }

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

    } while (frontier->count != 0);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
