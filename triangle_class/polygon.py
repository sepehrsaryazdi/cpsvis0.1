class PolygonVertex():
    def __init__(self, index):
        self.index = index
        self.edges = []

class PolygonEdge():
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1
        self.v0.edges.append(self)
        self.v1.edges.append(self)
        self.index = None
        self.edge_glued = None

class Polygon():
    def __init__(self, g, n):
        number_of_edges = 4*g
        self.vertices = []
        self.edges = []
        for i in range(number_of_edges):
            self.vertices.append(PolygonVertex(i))
        for i in range(number_of_edges):
            new_edge = PolygonEdge(self.vertices[i], self.vertices[(i+1)%number_of_edges])
            new_edge.index = i
            self.edges.append(new_edge)

    def glue_edges(self, edge, other_edge, initial_edge_vertex, initial_other_edge_vertex):
        if not edge.edge_glued:
            flipped = (initial_other_edge_vertex != other_edge.v0)
            edge.edge_glued = [initial_edge_vertex, initial_other_edge_vertex, other_edge]
            if flipped:
                other_edge.edge_glued = [other_edge.v0, edge.v1, edge]
            else:
                other_edge.edge_glued = [other_edge.v0, edge.v0, edge]