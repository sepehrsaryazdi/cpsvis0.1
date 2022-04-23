


from os import EX_CANTCREAT


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
        self.identification_index = None

class AbstractTriangle:
    def __init__(self, index):
        self.index = index
        self.vertices = [AbstractVertex(0), AbstractVertex(1),AbstractVertex(2)]
        self.edges = [AbstractEdge(self.vertices[0],self.vertices[1]),AbstractEdge(self.vertices[1],self.vertices[2]),AbstractEdge(self.vertices[2],self.vertices[0])]
        i=0
        for edge in self.edges:
            edge.triangle_edges_index = i
            edge.triangle = self
            i+=1
        self.selected = False

class AbstractSurface:
    def __init__(self):
        self.triangles = []

    def add_triangle(self):
        self.triangles.append(AbstractTriangle(len(self.triangles)))
        

    def glue_edges(self, edge, other_edge, initial_edge_vertex, initial_other_edge_vertex):
        
        flipped = (initial_other_edge_vertex != other_edge.v0)
        edge.edge_glued = [initial_edge_vertex, initial_other_edge_vertex, other_edge]
        if flipped:
            other_edge.edge_glued = [other_edge.v0, edge.v1, edge]
        else:
            other_edge.edge_glued = [other_edge.v0, edge.v0, edge]
    
    def flip_edge(self,edge):
        
        new_triangle = AbstractTriangle(edge.triangle.index)
        new_triangle_glued = AbstractTriangle(edge.edge_glued[2].triangle.index)

        edge_forward = edge.triangle.edges[(edge.triangle_edges_index+1)%3]
        edge_backward = edge.triangle.edges[(edge.triangle_edges_index-1)%3]
        edge_glued = edge.edge_glued[2]
        edge_glued_forward = edge_glued.triangle.edges[(edge_glued.triangle_edges_index+1)%3]
        edge_glued_backward = edge_glued.triangle.edges[(edge_glued.triangle_edges_index-1)%3]

        e_prime = new_triangle.edges[0]
        e_prime_connected = new_triangle_glued.edges[0]
        e_prime_forward = new_triangle.edges[1]
        e_prime_backward = new_triangle.edges[-1]
        e_prime_connected_forward = new_triangle_glued.edges[1]
        e_prime_connected_backward = new_triangle_glued.edges[-1]

        self.glue_edges(e_prime,e_prime_connected,e_prime.v0,e_prime_connected.v1)

        # new_triangle.edges[1] = edge.triangle.edges[(edge.triangle_edges_index-1)%3]
        # new_triangle.edges[1].triangle = new_triangle
        # new_triangle.edges[1].v0 = new_triangle.vertices[1]
        # new_triangle.edges[1].v1 = new_triangle.vertices[2]

        # new_triangle.edges[-1] = edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3]
        # new_triangle.edges[-1].triangle = new_triangle
        # new_triangle.edges[-1].v0 = new_triangle.vertices[2]
        # new_triangle.edges[-1].v1 = new_triangle.vertices[0]

        # new_triangle_glued.edges[1] = edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index-1)%3]
        # new_triangle_glued.edges[1].triangle = new_triangle_glued
        # new_triangle_glued.edges[1].v0 = new_triangle_glued.vertices[1]
        # new_triangle_glued.edges[1].v1 = new_triangle_glued.vertices[2]

        # new_triangle_glued.edges[-1] = edge.triangle.edges[(edge.triangle_edges_index+1)%3]
        # new_triangle_glued.edges[-1].triangle = new_triangle_glued
        # new_triangle_glued.edges[-1].v0 = new_triangle_glued.vertices[2]
        # new_triangle_glued.edges[-1].v1 = new_triangle_glued.vertices[0]

        flipped = (edge_backward.edge_glued[1] != edge_backward.edge_glued[2].v0)
        if not flipped:
            self.glue_edges(e_prime_forward, edge_backward.edge_glued[2], e_prime_forward.v0, edge_backward.edge_glued[2].v0)
        else:
            self.glue_edges(e_prime_forward, edge_backward.edge_glued[2], e_prime_forward.v0, edge_backward.edge_glued[2].v1)
        
        flipped = (edge_glued_forward.edge_glued[1] != edge_glued_forward.edge_glued[2].v0)
        if not flipped:
            self.glue_edges(e_prime_backward, edge_glued_forward.edge_glued[2], e_prime_backward.v0, edge_glued_backward.edge_glued[2].v0)
        else:
            self.glue_edges(e_prime_backward, edge_glued_forward.edge_glued[2], e_prime_backward.v0, edge_glued_backward.edge_glued[2].v1)
        
        flipped = (edge_glued_backward.edge_glued[1] != edge_glued_backward.edge_glued[2].v0)
        if not flipped:
            self.glue_edges(e_prime_connected_forward, edge_glued_backward.edge_glued[2], e_prime_connected_forward.v0, edge_glued_backward.edge_glued[2].v0)
        else:
            self.glue_edges(e_prime_connected_forward, edge_glued_backward.edge_glued[2], e_prime_connected_forward.v0, edge_glued_backward.edge_glued[2].v1)
        
        flipped = (edge_forward.edge_glued[1] != edge_forward.edge_glued[2].v0)
        if not flipped:
            self.glue_edges(e_prime_connected_backward, edge_forward.edge_glued[2], e_prime_connected_backward.v0, edge_forward.edge_glued[2].v0)
        else:
            self.glue_edges(e_prime_connected_backward, edge_forward.edge_glued[2], e_prime_connected_backward.v0, edge_forward.edge_glued[2].v1)
            
        
        # flipped = (edge.triangle.edges[(edge.triangle_edges_index-1)%3].edge_glued[1] != edge.triangle.edges[(edge.triangle_edges_index-1)%3].edge_glued[2].v0)
        # if not flipped:
        #     self.glue_edges(new_triangle.edges[1],edge.triangle.edges[(edge.triangle_edges_index-1)%3],new_triangle.edges[1].v0,edge.triangle.edges[(edge.triangle_edges_index-1)%3].v0)
        # else:
        #     self.glue_edges(new_triangle.edges[1],edge.triangle.edges[(edge.triangle_edges_index-1)%3],new_triangle.edges[1].v0,edge.triangle.edges[(edge.triangle_edges_index-1)%3].v1)
        # print(new_triangle.index,new_triangle.edges[1].index,new_triangle.edges[1].edge_glued[2].triangle.index,new_triangle.edges[1].edge_glued[2].index)
        # flipped = (edge.triangle.edges[(edge.triangle_edges_index+1)%3].edge_glued[1] != edge.triangle.edges[(edge.triangle_edges_index+1)%3].edge_glued[2].v0)
        # if not flipped:
        #     self.glue_edges(new_triangle_glued.edges[-1], edge.triangle.edges[(edge.triangle_edges_index+1)%3], new_triangle_glued.edges[-1].v0, edge.triangle.edges[(edge.triangle_edges_index+1)%3].v0)
        # else:
        #     self.glue_edges(new_triangle_glued.edges[-1], edge.triangle.edges[(edge.triangle_edges_index+1)%3], new_triangle_glued.edges[-1].v0, edge.triangle.edges[(edge.triangle_edges_index+1)%3].v1)
        
        # flipped = (edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3].edge_glued[1] != edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3].edge_glued[2].v0)
        # if not flipped:
        #     self.glue_edges(new_triangle.edges[-1], edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3],new_triangle.edges[-1].v0, edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3].v0)
        # else:
        #     self.glue_edges(new_triangle.edges[-1], edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3],new_triangle.edges[-1].v0, edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3].v1)

        # flipped = (edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index-1)%3].edge_glued[1] != edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index-1)%3].edge_glued[2].v0)
        # if not flipped:
        #     self.glue_edges(new_triangle_glued.edges[1], edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index-1)%3], new_triangle_glued.edges[1].v0,  edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index-1)%3].v0)
        # else:
        #     self.glue_edges(new_triangle_glued.edges[1], edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index-1)%3], new_triangle_glued.edges[1].v0,  edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index-1)%3].v1)
        
        
        self.triangles[edge.triangle.index] = new_triangle
        self.triangles[edge.edge_glued[2].triangle.index] = new_triangle_glued
        

        if len(edge.v0.coord):
            new_triangle.edges[0].v0.coord = edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3].v1.coord
            new_triangle.edges[0].v1.coord = edge.triangle.edges[(edge.triangle_edges_index-1)%3].v0.coord
            new_triangle.edges[1].v1.coord = edge.v0.coord
            new_triangle_glued.edges[1].v1.coord = edge.v1.coord
            new_triangle_glued.edges[0].v0.coord = new_triangle.edges[0].v1.coord
            new_triangle_glued.edges[0].v1.coord = new_triangle.edges[0].v0.coord
        
        try:
            new_triangle.edges[0].color = edge.color
            new_triangle.edges[0].arrow_strokes = edge.arrow_strokes
            new_triangle_glued.edges[0].color = edge.edge_glued[2].color
            new_triangle_glued.edges[0].arrow_strokes = edge.edge_glued[2].arrow_strokes
            new_triangle.edges[1].color = edge.triangle.edges[(edge.triangle_edges_index-1)%3].color
            new_triangle.edges[1].arrow_strokes = edge.triangle.edges[(edge.triangle_edges_index-1)%3].arrow_strokes
            new_triangle.edges[-1].color = edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3].color
            new_triangle.edges[-1].arrow_strokes = edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index+1)%3].arrow_strokes
            new_triangle_glued.edges[1].color = edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index-1)%3].color
            new_triangle_glued.edges[1].arrow_strokes = edge.edge_glued[2].triangle.edges[(edge.edge_glued[2].triangle_edges_index-1)%3].arrow_strokes
            new_triangle_glued.edges[-1].color = edge.triangle.edges[(edge.triangle_edges_index+1)%3].color
            new_triangle_glued.edges[-1].arrow_strokes = edge.triangle.edges[(edge.triangle_edges_index+1)%3].arrow_strokes
        except:
            pass


        return {e_prime_forward:edge_backward, e_prime_backward:edge_glued_forward, e_prime_connected_forward:edge_glued_backward, e_prime_connected_backward:edge_forward}

        

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



