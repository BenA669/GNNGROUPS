from manim import *
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def tensor_to_array(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()  # Convert tensor to NumPy array
    elif isinstance(data, (list, tuple)):
        return type(data)(tensor_to_array(x) for x in data)  # Recursively handle lists and tuples
    elif isinstance(data, dict):
        return {k: tensor_to_array(v) for k, v in data.items()}  # Recursively handle dictionaries
    else:
        return data  # Return as is if not a tensor or iterable
        
class GCNDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        (positions, adjacency, edge_indices, 
        ego_idx, ego_positions, ego_adjacency, 
        ego_edge_indices, EgoMask) = self.data[idx]
        # Convert everything to tensors
        return (positions, adjacency, edge_indices, 
                ego_idx, ego_positions, ego_adjacency, 
                ego_edge_indices, EgoMask)

def collate_fn(batch):
    # Unzip the batch (each sample is a tuple)
    positions, adjacency, edge_indices, ego_idx, ego_positions, ego_adjacency, ego_edge_indices, ego_mask = zip(*batch)
    B = len(edge_indices)
    T = len(edge_indices[0])    

    # print(ego_positions[0].shape)

    # ego_positions [batch, timestamp, node_amt, (x,y,Group)]
    
    # Stack the ego_positions (and any other elements you want to batch)
    positions_batch = torch.stack(positions, dim=0) # [batch_size, time_stamp, node_amt, 3]
    adjacency_batch = torch.stack(adjacency, dim=0) # [batchsize, time_stamp, node_amt, node_amt]
    ego_mask_batch = torch.stack(ego_mask, dim=0)

    max_nodes = positions_batch.size(dim=2)
    

    # [batchsize, time_stamp, node_amt, node_amt] -> [time_stamp, node_amt*batchsize, node_amt*batchsize]

    big_batch_edges = []
    big_batch_ego_edges = []
    big_batch_positions = []
    big_batch_adjacency = []
    # edge_indicies = [batch, timestamp, [2, N]]
    for t in range(T):
        edges_at_t = []
        ego_edges_at_t = []
        positions_at_t = []
        adjacency_at_t = []
        
        for b in range(B):
            # Get the edge-index for batch element b at timestamp t.
            e = edge_indices[b][t]  # shape [2, N_b]
            ee = ego_edge_indices[b][t]
            p = positions_batch[b, t, :, :2] # shape [node_amt, 2]
            a = adjacency_batch[b, t, :, :] # [batchsize, time_stamp, node_amt, node_amt] -> [node_amt, node_amt]

            # Offset the node indices so that nodes in batch b get indices in [b*max_nodes, (b+1)*max_nodes)
            e_offset = e + b * max_nodes
            ee_offset = ee + b * max_nodes
            # p_offset = p + b*max_nodes # The positions don't need to be increased? I ws not cooking
            
            edges_at_t.append(e_offset)
            ego_edges_at_t.append(ee_offset)
            # positions_at_t.append(p_offset)
            positions_at_t.append(p)
            adjacency_at_t.append(a)

        # Concatenate all batches’ edge indices for this timestamp along the second dimension.
        combined_edges = torch.cat(edges_at_t, dim=1).to(torch.device('cuda:0'))  # shape [2, total_edges_at_t]
        big_batch_edges.append(combined_edges)

        combined_ego_edges = torch.cat(ego_edges_at_t, dim=1).to(torch.device('cuda:0'))
        big_batch_ego_edges.append(combined_ego_edges)

        combined_pos = torch.cat(positions_at_t, dim=0).to(torch.device('cuda:0'))
        big_batch_positions.append(combined_pos)

        stacked_adj = torch.stack(adjacency_at_t, dim=0)
        combined_adj = torch.block_diag(*stacked_adj)
        big_batch_adjacency.append(combined_adj)
        

    big_batch_positions = torch.stack(big_batch_positions, dim=0).to(torch.device('cuda:0'))
    big_batch_adjacency = torch.stack(big_batch_adjacency, dim=0).to(torch.device('cuda:0'))
    # print(type(big_batch_edges[0]))
    # print(len(big_batch_edges[1][0]))
    # print(big_batch_adjacency.shape)
    
    # You can also stack or combine the other items if needed.
    # For now, we'll return all components in a dictionary:
    return {
        'positions': positions_batch,
        'adjacency': adjacency,
        'edge_indices': edge_indices,
        'ego_idx': ego_idx,
        'ego_positions': ego_positions,  # This now has shape [batch, timesteps, node_amt, 3]
        'ego_adjacency': ego_adjacency,
        'ego_edge_indices': ego_edge_indices,
        'ego_mask_batch': ego_mask_batch,
        'big_batch_edges': big_batch_edges,
        'big_batch_positions': big_batch_positions,
        'big_batch_ego_edges': big_batch_ego_edges,
        'big_batch_adjacency': big_batch_adjacency,
    }

class HELP(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()
        # Create dataset
        # dataset = GCNDataset('test_data_Ego.pt')
        timestamp=7

        # Create DataLoader
        # dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
        dataloader = range(5)
        for batch_idx, batch in enumerate(dataloader):
            # big_batch_positions = batch['big_batch_positions']
            # subset = np.around(tensor_to_array(big_batch_positions[0, :10, :]) , decimals=1) 

            subset=np.around([[ 0.3625,  2.0133],
                    [ 1.0535, -0.0364],
                    [-1.3407,  0.1890],
                    [-2.0156,  0.0806],
                    [-2.2781,  1.3505],
                    [ 1.6718,  0.4890],
                    [ 1.3685, -0.3850],
                    [-0.1701, -0.4103],
                    [ 1.1898, -1.2578],
                    [ 2.9167,  1.1061]], decimals=1)
            # [Timestamp, Node, 2]
            
            featMatrix = Matrix(subset)
            

            featMatrix.scale(0.5)
            tFeatMatrix = Text(f"Feature Matrix").next_to(featMatrix, UP)

            self.play(Write(featMatrix), Write(tFeatMatrix))

            feat_brace = Brace(featMatrix)
            tFeat = Text(f"Node Features").next_to(feat_brace, DOWN, buff=-0.1).scale(0.5)

            node_brace = Brace(featMatrix, direction=[-1, 0, 0])
            tNode = Text(f"Node Amount").next_to(node_brace, LEFT, buff=-1.8).scale(0.5).rotate(np.pi/2)
            self.play(Write(node_brace), Write(tNode), Write(feat_brace), Write(tFeat))

            # group1 = Group(featMatrix, )

            self.wait(1)
            self.play(FadeOut(feat_brace, tFeat, node_brace, tNode))
            tFeatMatrixTime = Text(f"Feature Matrix over Episode").next_to(featMatrix, UP)
            self.play(featMatrix.animate.move_to(LEFT*5.5))

            # featBoxes = []
            # for ent in featMatrix.get_rows():
            #     featBoxes.append(SurroundingRectangle(ent, buff=0.07))

            featMatricies = []
            featMatricies.append(featMatrix)
            featMatriciesAnimations = []
            featMatriciesAnimations.append(FadeOut(tFeatMatrix))
            # featBoxesBig = []
            # featBoxesBig.append(featBoxes)
            for i in range(1, timestamp):
                # print(i)
                featMatricies.append(featMatrix.copy().next_to(featMatricies[i-1], RIGHT))
                featMatriciesAnimations.append(FadeIn(featMatricies[i], shift=RIGHT))
                # featBoxes = []
                # for ent in featMatricies[i].get_rows():
                #     featBoxes.append(SurroundingRectangle(ent, buff=0.07))
                # featBoxesBig.append(featBoxes)
            featMatriciesAnimations.append(Write(tFeatMatrixTime))
            bigMatrixFeat = Group(*featMatricies)

            time_brace = Brace(bigMatrixFeat, direction=DOWN)
            tTime = Text(f"Timesteps").next_to(time_brace, DOWN).scale(0.5)
            featMatriciesAnimations.append(Write(time_brace))
            featMatriciesAnimations.append(Write(tTime))


            # featMatrix2 = featMatrix.copy().next_to(featMatrix, RIGHT)
            # self.play(FadeIn(featMatrix2, shift=RIGHT))
            self.play(AnimationGroup(*featMatriciesAnimations, lag_ratio=0.1))
            
            egoMask = [[True, False, True, True, False, False, False, True, True, False]]
            egoMask2 = [[False, False, False, True, False, True, False, False, True, True]]

            egoMask_T = [["T"], [ "F"], [ "T"], [ "T"], [ "F"], [ "F"], [ "F"], [ "T"], [ "T"], [ "F"]]
            egoMask2_T = [["F"], [ "F"], [ "F"], [ "T"], [ "F"], [ "T"], [ "F"], [ "F"], [ "T"], [ "T"]]


            self.wait(1)

            self.play(FadeOut(time_brace, tTime))
            
            bigEgoMaskMatrix = []
            for i in range(timestamp):
                if i > 3:
                    egoMaskMatrix = Matrix(egoMask, h_buff=1.5).scale(0.5).next_to(bigMatrixFeat, DOWN)     
                else:
                    egoMaskMatrix = Matrix(egoMask2, h_buff=1.5).scale(0.5).next_to(bigMatrixFeat, DOWN)     
                    
                egoMaskMatrixEnt = egoMaskMatrix.get_entries()
                for idx, ent in enumerate(egoMaskMatrixEnt):
                    if egoMask2[0][idx] == True:
                        ent.set_color(GREEN)
                    else:
                        ent.set_color(RED)
                bigEgoMaskMatrix.append(egoMaskMatrix)

            tEgoMaskMatrix = Text(f"Ego Mask Matrix at t=0").scale(0.5).next_to(egoMaskMatrix, DOWN)
            self.play(Write(bigEgoMaskMatrix[0]), Write(tEgoMaskMatrix))
            
            self.wait(1)

            # Make new TF matrix next to corresponding feat matrix
            # Replacement transform both feat matrix and ego matrix. (Feat matrix to make room)
            transposedEgoMaskMatrix = []
            movedFeatMatricies = []
            tEMMAnimation = []
            mFMAnimation = []
            featBoxesBig = []
            for i in range(timestamp):
                featBoxes = []
                if i > 3:
                    transposedEgoMaskMatrix.append(Matrix(egoMask_T).scale(0.5).next_to(movedFeatMatricies[i-1], RIGHT*2))
                    movedFeatMatricies.append(featMatricies[i].copy().next_to(transposedEgoMaskMatrix[i], RIGHT))

                    for idx, ent in enumerate(transposedEgoMaskMatrix[i].get_rows()):
                        if egoMask[0][idx] == True:
                            ent.set_color(GREEN)
                        else:
                            ent.set_color(RED)
                        localPair = VGroup(transposedEgoMaskMatrix[i], movedFeatMatricies[i])
                        featBoxes.append(SurroundingRectangle(ent, buff=0.07).match_x(localPair).stretch_to_fit_width(2.5))
                    featBoxesBig.append(featBoxes)

                else:
                    if i == 0:
                        transposedEgoMaskMatrix.append(Matrix(egoMask2_T).scale(0.5).next_to(featMatricies[i], LEFT))
                        movedFeatMatricies.append(featMatricies[i].copy().next_to(transposedEgoMaskMatrix[i], RIGHT))
                        tEMMAnimation.append(ReplacementTransform(bigEgoMaskMatrix[i], transposedEgoMaskMatrix[i]))
                    else:
                        transposedEgoMaskMatrix.append(Matrix(egoMask2_T).scale(0.5).next_to(movedFeatMatricies[i-1], RIGHT*2))
                        movedFeatMatricies.append(featMatricies[i].copy().next_to(transposedEgoMaskMatrix[i], RIGHT))
                    for idx, ent in enumerate(transposedEgoMaskMatrix[i].get_rows()):
                        if egoMask2[0][idx] == True:
                            ent.set_color(GREEN)
                        else:
                            ent.set_color(RED)
                        localPair = VGroup(transposedEgoMaskMatrix[i], movedFeatMatricies[i])
                        featBoxes.append(SurroundingRectangle(ent, buff=0.07).match_x(localPair).stretch_to_fit_width(2.5))
                    featBoxesBig.append(featBoxes)
                
                

                if i != 0:
                        tEMMAnimation.append(FadeIn(transposedEgoMaskMatrix[i], shift=RIGHT))
                mFMAnimation.append(ReplacementTransform(featMatricies[i], movedFeatMatricies[i]))
            
            # Group them together to compute their combined bounding box
            all_mobjects = VGroup(*movedFeatMatricies)

            # Compute a little extra buffer (e.g., 20% more)
            buffer = 1.2

            # Animate the camera frame to center on the group and adjust its width.
            self.play(
                self.camera.frame.animate.move_to(all_mobjects.get_center()).set(width=all_mobjects.get_width() * buffer)
            )

            tMaskFeatMatrix = Text(f"Masked Feature Matrix").next_to(all_mobjects, UP)
            self.play(FadeOut(tEgoMaskMatrix), FadeOut(tFeatMatrixTime))
            self.play(AnimationGroup(*mFMAnimation, lag_ratio=0.07))
            self.play(AnimationGroup(*tEMMAnimation, lag_ratio=0.1), Write(tMaskFeatMatrix))

            
            # boxEgo = SurroundingRectangle(egoMaskMatrix.get_columns()[0], buff=0.07)

            # self.play(Create(boxEgo))
            startBoxCreate = []
            # self.play(Create(boxEgos[0]))
            for featBoxes in featBoxesBig:
                startBoxCreate.append(Create(featBoxes[0]))
            self.play(AnimationGroup(*startBoxCreate, lag_ratio=0.1))
            # self.play(Succession(*boxEgosAnimation))
            for idx in range(0, len(featBoxes)):
                iterAnimation = []
                if idx == 0:
                    for idxLINE, featBoxes in enumerate(featBoxesBig):
                        if idxLINE > 3:
                            if egoMask[0][idx] == True:
                                setColor = GREEN
                            else:
                                setColor = RED
                        else:
                            if egoMask2[0][idx] == True:
                                setColor = GREEN
                            else:
                                setColor = RED
                        iterAnimation.append(movedFeatMatricies[idxLINE].get_rows()[idx].animate(run_time=0.25).set_color(setColor))
                else:
                    for idxLINE, featBoxes in enumerate(featBoxesBig):
                        # self.play(ReplacementTransform(featBoxes[idx-1], featBoxes[idx]))
                        if idxLINE > 3:
                            if egoMask[0][idx] == True:
                                setColor = GREEN
                            else:
                                setColor = RED
                        else:
                            if egoMask2[0][idx] == True:
                                setColor = GREEN
                            else:
                                setColor = RED
                        iterAnimation.append(movedFeatMatricies[idxLINE].get_rows()[idx].animate(run_time=0.15).set_color(setColor))
                        iterAnimation.append(ReplacementTransform(featBoxes[idx-1], featBoxes[idx], run_time=0.15))
                        

                self.play(AnimationGroup(*iterAnimation, lag_ratio=0.1))

            byboxa = []
            for featboxa in featBoxesBig:
                byboxa.append(FadeOut(featboxa[-1]))
            
            self.play(AnimationGroup(*byboxa))


            subsetMASK1=np.around(
                    [[ 0.3625,  2.0133],
                    [-1.3407,  0.1890],
                    [-2.0156,  0.0806],
                    [-0.1701, -0.4103],
                    [ 1.1898, -1.2578],], decimals=1)
            
            subsetMASK2=np.around(
                    [[-2.0156,  0.0806],
                    [ 1.6718,  0.4890],
                    [ 1.1898, -1.2578],
                    [ 2.9167,  1.1061]], decimals=1)
            
            # transposedEgoMaskMatrix
            # movedFeatMatricies
            afterMaskAnimation = []
            maskedMatricies = []
            for i in range(0, timestamp):
                if i > 3:
                    maskedMatricies.append(Matrix(subsetMASK1).scale(0.5))
                else:
                    maskedMatricies.append(Matrix(subsetMASK2).scale(0.5))
                # before = VGroup(movedFeatMatricies[i], transposedEgoMaskMatrix[i])
                before = movedFeatMatricies[i]
                # if i == 0:
                #     afterMaskAnimation.append(ReplacementTransform(before, maskedMatricies[i].move_to(LEFT*5.5)))
                # else:
                afterMaskAnimation.append(ReplacementTransform(before, maskedMatricies[i].match_x(movedFeatMatricies[i])))
                afterMaskAnimation.append(FadeOut(transposedEgoMaskMatrix[i]))

            self.play(AnimationGroup(*afterMaskAnimation))
            self.wait(5)
            break


    