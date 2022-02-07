import numpy as np
#
# class Decoration:
#     def __init__(self, s0, s1, s2, r0, r1, r2):
#         self.s0 = s0
#         self.s1 = s1
#         self.s2 = s2
#         self.r0 = r0
#         self.r1 = r1
#         self.r2 = r2
#         assert len(s0) == 3, "s0 not a valid R^3 vector"
#         assert len(s1) == 3, "s1 not a valid R^3 vector"
#         assert len(s2) == 3, "s0 not a valid R^3 vector"
#
#     def normalise_decoration(self):
#         self.s0 = self.s0/np.linalg.norm(self.s0)
#         self.s1 = self.s1/np.linalg.norm(self.s1)
#         self.s2 = self.s2/np.linalg.norm(self.s2)

class Vertex:
    def __init__(self,c,r, c_clover, r_clover):
        self.c = c
        self.r = r
        self.c_clover = c_clover
        self.r_clover = r_clover
        self.edges = []


class Edge:
    def __init__(self, v0,v1, connected):
        self.v0 = v0
        self.v1 = v1
        self.v0.edges.append(self)
        self.v1.edges.append(self)
        self.ev0v1 = np.dot(self.v0.r,self.v1.c)
        self.ev1v0 = np.dot(self.v1.r, self.v0.c)
        self.triangle = []
        self.connected = False
        self.edge_connected = None

class Triangle:
    def __init__(self, e0, e1, e2):
        self.edges = [e0, e1, e2]
        self.vertices = [e0.v0, e1.v0, e2.v0]
        edge_index = 0
        for edge in self.edges:
            edge.triangle = self
            edge.index = edge_index
            edge_index+=1
        [v0, v1, v2] = self.vertices

        #print([v.c for v in self.vertices])
        # if np.linalg.det([v0.c,v1.c,v2.c]) < 0:
        #     self.vertices = [v0, v2, v1]
        # if np.linalg.det([v0.c,v1.c,v2.c]) < 0:
        #     for edge in self.edges:
        #         [edge.v0,edge.v1] = [edge.v1, edge.v0]

        #print(np.linalg.det([self.edges[0].v0.c, self.edges[1].v0.c, self.edges[2].v0.c]))

        #print(self.t)
        self.neighbours = []
    def add_neighbour(self, neighbour_triangle):
        self.neighbours.append(neighbour_triangle)

class Surface:
    def __init__(self, c0, c1, c2, r0, r1, r2, c0_clover, c1_clover, c2_clover, r0_clover, r1_clover, r2_clover):
        vertices = [Vertex(c0,r0, c0_clover, r0_clover), Vertex(c1,r1, c1_clover, r1_clover), Vertex(c2,r2, c2_clover, r2_clover)]
        edges = [Edge(vertices[0],vertices[1], False),Edge(vertices[1],vertices[2], False),Edge(vertices[2],vertices[0], False)]
        #print(np.linalg.det([c0,c1,c2]))
        initial_triangle = Triangle(edges[0],edges[1],edges[2])
        self.triangles = [initial_triangle]
        initial_triangle.distance_from_centre = 0
        initial_triangle.index = 0
    def add_triangle(self, connecting_edge, v0, v1, new_vertex):
        # if np.linalg.det(np.array([v0.c,v1.c,new_vertex.c])) > 0:
        #     print(v0.c_clover,v1.c_clover, new_vertex.c_clover)

        connecting_edge.connected = True
        new_triangle = Triangle(Edge(v0, v1, True), Edge(v1,new_vertex, False), Edge(new_vertex,v0, False))
        #print([v.c_clover for v in new_triangle.vertices])
        connecting_edge.edge_connected = new_triangle.edges[0]
        new_triangle.edges[0].edge_connected = connecting_edge
        new_triangle.edges[0].connected = True
        self.triangles.append(new_triangle)
        new_triangle.add_neighbour(connecting_edge.triangle)
        self.triangles[-1].add_neighbour(new_triangle)
        new_triangle.index = connecting_edge.triangle.index+1
        new_triangle.distance_from_centre = connecting_edge.triangle.distance_from_centre+1
        return new_triangle

    def connect_edges(self, e1,e2):
        e1.edge_connected = e2
        e2.edge_connected = e1
        e1.connected = True
        e2.connected = True

    def flip_edge(self, edge):

        edge_forward = edge.triangle.edges[(edge.index+1) % 3]
        edge_backward = edge.triangle.edges[(edge.index-1) % 3]
        edge_connected = edge.edge_connected
        edge_connected_forward = edge_connected.triangle.edges[(edge_connected.index+1) % 3]
        edge_connected_backward = edge_connected.triangle.edges[(edge_connected.index - 1)%3]
        e_prime = Edge(edge_forward.v1, edge_connected_backward.v0, True)
        e_prime_forward = edge_connected_backward
        e_prime_backward = edge_forward
        e_prime_connected = Edge(e_prime.v1, e_prime.v0, e_prime.connected)
        self.connect_edges(e_prime,e_prime_connected)
        e_prime_connected_forward = edge_backward
        e_prime_connected_backward = edge_connected_forward
        triangle_1 = edge.triangle
        triangle_2 = edge_connected.triangle

        for triangle_index in range(len(self.triangles)):
            if self.triangles[triangle_index] == triangle_1:
                new_triangle_1 = Triangle(e_prime, e_prime_forward, e_prime_backward)
                new_triangle_1.index = triangle_1.index
                self.triangles[triangle_index] = new_triangle_1
                for neighbour_index in range(len(self.triangles[triangle_index].neighbours)):
                    if self.triangles[triangle_index].neighbours[neighbour_index] == triangle_1:
                        self.triangles[triangle_index].neighbours[neighbour_index] = self.triangles[triangle_index]
                for neighbour in triangle_1.neighbours:
                    self.triangles[triangle_index].neighbours.append(neighbour)

        for triangle_index in range(len(self.triangles)):
            if self.triangles[triangle_index] == triangle_2:
                new_triangle_2 = Triangle(e_prime_connected, e_prime_connected_forward, e_prime_connected_backward)
                new_triangle_2.index = triangle_2.index
                self.triangles[triangle_index] = new_triangle_2
                for neighbour_index in range(len(self.triangles[triangle_index].neighbours)):
                    if self.triangles[triangle_index].neighbours[neighbour_index] == triangle_2:
                        self.triangles[triangle_index].neighbours[neighbour_index] = self.triangles[triangle_index]
                for neighbour in triangle_2.neighbours:
                    self.triangles[triangle_index].neighbours.append(neighbour)




    def normalise_vertices(self):
        all_vertices = []
        for triangle in self.triangles:
            for vertex in triangle.vertices:
                if vertex not in all_vertices:
                    all_vertices.append(vertex)
        for vertex in all_vertices:
            vertex.c = [vertex.c[0],vertex.c[1],vertex.c[2],1]
            #vertex.c_clover = [vertex.c_clover[0], vertex.c_clover[1], vertex.c_clover[2], 1]
            vertex.c = vertex.c/np.linalg.norm(vertex.c)
            #vertex.c_clover = vertex.c_clover / np.linalg.norm(vertex.c_clover)
            vertex.c = np.array(vertex.c[1:])/(1+vertex.c[0])
            #vertex.c_clover = np.array(vertex.c_clover[1:])/(1+vertex.c_clover[0])

    # def add_vertex(self, triangle, new_vertex):
    #     decoration = triangle.decoration
    #     decoration = np.array([np.transpose(decoration.s0), np.transpose(decoration.s1), np.transpose(decoration.s2)])
    #     distances = np.linalg.norm(np.repeat([new_vertex],3,axis=0)-decoration,axis=1)
    #     other_vertices = decoration[np.argsort(distances)[:2]]
    #     determinant = np.linalg.det([other_vertices[0],other_vertices[1],new_vertex])
    #     assert determinant != 0, 'New Vertex does not span a triangle.'
    #     if determinant > 0:
    #         self.add_triangle(triangle,Triangle(Decoration(other_vertices[0],other_vertices[1],new_vertex)))
    #     else:
    #         self.add_triangle(triangle, Triangle(Decoration(other_vertices[0],new_vertex,other_vertices[1])))



