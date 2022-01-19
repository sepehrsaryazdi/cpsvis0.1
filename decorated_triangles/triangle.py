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
    def __init__(self,c,r):
        self.c = c
        self.r = r


class Edge:
    def __init__(self, v0,v1):
        self.v0 = v0
        self.v1 = v1
        self.v0.edge = self
        self.v1.edge = self

class Triangle:
    def __init__(self, e0, e1, e2):
        self.edges = [e0, e1, e2]
        for edge in self.edges:
            edge.triangle = self
        self.neighbours = []
    def add_neighbour(self, neighbour_triangle):
        self.neighbours.append(neighbour_triangle)

class Surface:
    def __init__(self, c0, c1, c2, r0, r1, r2):
        vertices = [Vertex(c0,r0), Vertex(c1,r1), Vertex(c2,r2)]
        edges = [Edge(vertices[0],vertices[1]),Edge(vertices[1],vertices[2]),Edge(vertices[2],vertices[0])]
        initial_triangle = Triangle(edges[0],edges[1],edges[2])
        self.triangles = [initial_triangle]
    def add_triangle(self, connecting_edge, new_vertex):
        new_triangle = Triangle(connecting_edge, Edge(connecting_edge.v1,new_vertex), Edge(new_vertex,connecting_edge.v0))
        self.triangles.append(connecting_edge.triangle)
        new_triangle.add_neighbour(self.triangles[-1])
        self.triangles[-1].add_neighbour(new_triangle)
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



