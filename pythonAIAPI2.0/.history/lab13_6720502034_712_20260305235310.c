//6720502034 thanakorn promlek g.712

#include <stdio.h>

int main() {
    int num_vertices, num_edges;

    printf("Enter number of vertices: ");
    scanf("%d", &num_vertices);

    printf("Enter number of edges: ");
    scanf("%d", &num_edges);

    printf("Enter the edges (u v) format:\n");

    int u, v;
    int adj_matrix[num_vertices][num_vertices];
    int is_visited[num_vertices];

    for (int i = 0; i < num_vertices; i++) {
        is_visited[i] = 0;
        for (int j = 0; j < num_vertices; j++) {
            adj_matrix[i][j] = 0;
        }
    }

    for (int i = 0; i < num_edges; i++) {
        scanf("%d %d", &u, &v);
        adj_matrix[u][v] = 1;
        adj_matrix[v][u] = 1; 
    }

    printf("Approximate Vertex Cover: ");
    
    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < num_vertices; j++) {
            if (adj_matrix[i][j] == 1 && is_visited[i] == 0 && is_visited[j] == 0) {
                is_visited[i] = 1;
                is_visited[j] = 1;
                printf("(%d,%d) ", i, j);
            }
        }
    }

    return 0;
}