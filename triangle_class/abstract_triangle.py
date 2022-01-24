


class Edge:
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1
        self.v0.edges.append(self)
        self.v1.edges.append(self)
        self.triangle = None
        self.edge_glued = None
        self.index = f'{v0.index}{v1.index}'


class Vertex:
    def __init__(self, index):
        self.index = index
        self.edges = []

class Triangle:
    def __init__(self, index):
        self.index = index
        self.vertices = [Vertex(0), Vertex(1),Vertex(2)]
        self.edges = [Edge(self.vertices[0],self.vertices[1]),Edge(self.vertices[1],self.vertices[2]),Edge(self.vertices[0],self.vertices[2])]
        for edge in self.edges:
            edge.triangle = self

class AbstractSurface:
    def __init__(self):
        self.triangles = []

    def add_triangle(self):
        self.triangles.append(Triangle(len(self.triangles)))

    def glue_edges(self, edge, other_edge, initial_edge_vertex, initial_other_edge_vertex):
        if not edge.edge_glued:
            edge.edge_glued = [initial_edge_vertex, initial_other_edge_vertex, other_edge]
            other_edge.edge_glued = [initial_other_edge_vertex, initial_edge_vertex, edge]

    def give_vertex_coordinates(self, vertex, coord):
        try:
            vertex.coord
        except:
            vertex.coord = coord

            for edge in vertex.edges:
                if edge.edge_glued:
                    flipped = (edge.edge_glued[1] != edge.edge_glued[2].v0)
                    vertex_is_at_end_of_edge = (edge.v1 == vertex)
                    if not flipped:
                        if vertex_is_at_end_of_edge:
                            other_vertex = edge.edge_glued[2].v1
                        else:
                            other_vertex = edge.edge_glued[2].v0
                    else:
                        if vertex_is_at_end_of_edge:
                            other_vertex = edge.edge_glued[2].v0
                        else:
                            other_vertex = edge.edge_glued[2].v1
                    self.give_vertex_coordinates(other_vertex, coord)
                    #other_vertex.coord = coord



