


class AbstractEdge:
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1
        self.v0.edges.append(self)
        self.v1.edges.append(self)
        self.triangle = None
        self.edge_glued = None
        self.index = f'{v0.index}{v1.index}'


class AbstractVertex:
    def __init__(self, index):
        self.index = index
        self.edges = []
        self.coord = []

class AbstractTriangle:
    def __init__(self, index):
        self.index = index
        self.vertices = [AbstractVertex(0), AbstractVertex(1),AbstractVertex(2)]
        self.edges = [AbstractEdge(self.vertices[0],self.vertices[1]),AbstractEdge(self.vertices[1],self.vertices[2]),AbstractEdge(self.vertices[0],self.vertices[2])]
        for edge in self.edges:
            edge.triangle = self
        self.selected = False

class AbstractSurface:
    def __init__(self):
        self.triangles = []

    def add_triangle(self):
        self.triangles.append(AbstractTriangle(len(self.triangles)))

    def glue_edges(self, edge, other_edge, initial_edge_vertex, initial_other_edge_vertex):
        if not edge.edge_glued:
            flipped = (initial_other_edge_vertex != other_edge.v0)
            edge.edge_glued = [initial_edge_vertex, initial_other_edge_vertex, other_edge]
            if flipped:
                other_edge.edge_glued = [other_edge.v0, edge.v1, edge]
            else:
                other_edge.edge_glued = [other_edge.v0, edge.v0, edge]

    def give_vertex_coordinates(self, vertex, coord):
        if not len(vertex.coord):
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



