from manim import *
from BestEvalREAL import getData
from torch import torch

class MovingVertices(Scene):
    def construct(self):
        positions, adjacency, edge_indices, ego_idx, ego_mask, ego_positions = getData()

        vertices = [1, 2, 3, 4]
        edges = [(1, 2), (2, 3), (3, 4), (1, 3), (1, 4)]

        


        g = Graph(vertices, edges)
        self.play(Create(g))
        self.wait()
        self.play(g[1].animate.move_to([1, 1, 0]),
                  g[2].animate.move_to([-1, 1, 0]),
                  g[3].animate.move_to([1, -1, 0]),
                  g[4].animate.move_to([-1, -1, 0]))
        self.wait()

class GraphManualPosition(Scene):
    def construct(self):
        positions, adjacency, edge_indices, ego_idx, ego_mask, ego_positions = getData()
        print("Get Data Done...")

        node_amt = positions.size(dim=1)
        print("Node amount: {}".format(node_amt))
        # vertices = [1, 2, 3, 4]
        vertices = torch.arange(0,node_amt).flatten().tolist()

        permute_edge = edge_indices[0].permute(1, 0)
        # edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        edges = list(map(tuple, permute_edge.tolist()))
        print(permute_edge.shape)  # Should be (N, 2)
        print(edges[:5])  # Check first 5 edges


        # positions = [time, node, 3]
        positions[:, :, 2] = 0
        
        positions = positions[0, :, :]*0.2

        print(positions[:5])
        lt = {i: sublist for i, sublist in enumerate(positions.cpu().tolist())}
        # lt = {1: [0, 0, 0], 2: [1, 1, 0], 3: [1, -1, 4], 4: [-1, 0, 0]} , layout=lt

        g = Graph(vertices, edges, layout=lt, edge_config={"stroke_width": 0.4})
        # self.add(g)
        self.play(Create(g))
        self.wait(2)
        # self.play(Create(g))
        # self.wait()
        # self.play(g[1].animate.move_to([1, 1, 0]),
        #           g[2].animate.move_to([-1, 1, 0]),
        #           g[3].animate.move_to([1, -1, 0]),
        #           g[4].animate.move_to([-1, -1, 0]))
        # self.wait()

# if __name__ == '__main__':
#     positions, adjacency, edge_indices, ego_idx, ego_mask, ego_positions = getData()
#     print("getData Done")
#     print(positions.shape)
#     print(adjacency.shape)