#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include<cmath>
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
        #pragma omp simd
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

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    #pragma omp parallel
    {
    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        #pragma omp single
        {
            vertex_set_clear(new_frontier);
        }
        #pragma omp barrier
        #pragma omp single
        {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        }
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        #pragma omp single
        {
            vertex_set *tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        }
    }
    }
}

vertex_set* bottom_up_step(Graph g, int *distances, int curDis, vertex_set *new_frontier) {
    #pragma omp parallel 
    {
        #pragma omp for nowait
        for (int i = 0; i < g->num_nodes; i++) {
            if (distances[i] != NOT_VISITED_MARKER) 
                continue;
            
            const Vertex* start = incoming_begin(g, i);
            const Vertex* end = incoming_end(g, i);
            bool found = 0;
            for (const Vertex* neighbor = start; neighbor != end; neighbor++) {
                if (distances[*neighbor] == curDis) {
                    if (new_frontier) {
                        int index = __sync_fetch_and_add(&new_frontier->count, 1);
                        new_frontier->vertices[index] = i;
                    }
                    distances[i] = curDis + 1;
                    break;
                }
            }
        }
    }
    return new_frontier;
}

void bfs_bottom_up(Graph graph, solution *sol) {
    
    #pragma omp parallel for simd
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = -1;
    }

    sol->distances[ROOT_NODE_ID] = 0;
    int curDis = 0;
    int remainCount = graph->num_nodes - 1;

    while (1) {
        int preCount = remainCount;

        bottom_up_step(graph, sol->distances, curDis, nullptr);

        remainCount = 0;
        #pragma omp parallel for reduction(+:remainCount)
        for (int i = 0; i < graph->num_nodes; i++) {
            if (sol->distances[i] == -1) remainCount++;
        }

        if (preCount == remainCount) 
            break;
        
        curDis++;
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    int numNodes = graph -> num_nodes;
    int threshold  = static_cast <int> (round(sqrt( static_cast <float>(numNodes))))/10;

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int curDis = 0;
    int remainCount = graph->num_nodes - 1;

    while (frontier->count != 0) {
        bool useBottomUp = frontier->count > threshold;
        vertex_set_clear(new_frontier);

        if (useBottomUp) {
            int preCount = 0;
            int remainCount = 0;

            #pragma omp parallel for reduction(+:preCount)
            for (int i = 0; i < graph->num_nodes; i++) 
                if (sol->distances[i] == -1) preCount++;
            
            bottom_up_step(graph, sol -> distances, curDis, new_frontier);

            #pragma omp parallel for reduction(+:remainCount)
            for (int i = 0; i < graph->num_nodes; i++) 
                if (sol->distances[i] == -1) remainCount++;

            if (preCount == remainCount) 
                break;
        } else {
            top_down_step(graph, frontier, new_frontier, sol -> distances);
        }
        curDis ++;
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
