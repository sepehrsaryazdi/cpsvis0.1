import numpy as np

class Decoration:
    def __init__(self, s0, s1, s2):
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        assert len(s0) == 3, "s0 not a valid R^3 vector"
        assert len(s1) == 3, "s1 not a valid R^3 vector"
        assert len(s2) == 3, "s0 not a valid R^3 vector"

    def normalise_decoration(self):
        self.s0 = self.s0/np.linalg.norm(self.s0)
        self.s1 = self.s1/np.linalg.norm(self.s1)
        self.s2 = self.s2/np.linalg.norm(self.s2)

class Triangle:
    def __init__(self, decoration):
        self.decoration = decoration
        self.neighbours = []
    def update_decoration(self,decoration):
        self.decoration = decoration
    def add_neighbour(self, neighbour_triangle):
        self.neighbours.append(neighbour_triangle)
    def remove_neighbour(self, neighbour_triangle):
        self.neighbours.remove(neighbour_triangle)

class Surface:
    def __init__(self, genus, punctures, initial_triangle):
        self.triangles = [initial_triangle]
        self.g = genus
        self.n = punctures
    def add_triangle(self, start_triangle, new_triangle):
        self.triangles.append(new_triangle)
        start_triangle.add_neighbour(self.triangles[-1])
        self.triangles[-1].add_neighbour(start_triangle)
    def remove_triangle(self, triangle):
        self.triangles.remove(triangle)
        for neighbour in triangle.neighbours:
            neighbour.remove_neighbours(triangle)
        for neighbour in triangle.neighbours.copy():
            triangle.remove_neighbour(neighbour)
    def add_vertex(self, triangle, new_vertex):
        decoration = triangle.decoration
        decoration = np.array([np.transpose(decoration.s0), np.transpose(decoration.s1), np.transpose(decoration.s2)])
        distances = np.linalg.norm(np.repeat([new_vertex],3,axis=0)-decoration,axis=1)
        other_vertices = decoration[np.argsort(distances)[:2]]
        #self.add_triangle()
        determinant = np.linalg.det([other_vertices[0],other_vertices[1],new_vertex])
        assert determinant != 0, 'New Vertex does not span a triangle.'
        if determinant > 0:
            self.add_triangle(triangle,Triangle(Decoration(other_vertices[0],other_vertices[1],new_vertex)))
        else:
            self.add_triangle(triangle, Triangle(Decoration(other_vertices[0],new_vertex,other_vertices[1])))



