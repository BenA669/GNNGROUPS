from manim import *
import numpy as np
from MannDynamicVis import displayAllMatrix, titleObj, indexMatrix, braceMatrix, colorListofMatrices, labelTimestamps, makeSurroundRectList, zipperMatrix, maskMatrix, gcnPass, lstmpass
# All your helper functions (displayAllMatrix, braceMatrix, colorMatrix, etc.)
# can be defined here (or imported from another module).

####################################
# Scene 1: Feature and Adjacency Matrices
####################################
class FeatureAdjacencyScene(MovingCameraScene):
    def construct(self):
        np.random.seed(0)
        num_nodes = 7
        timestamps = 3
        node_features = 2
        # ... (other parameters)

        # Create your matrices (all_node_features, all_adj_matrix, etc.)
        all_node_features = np.round(np.random.rand(timestamps, num_nodes, node_features)*100, decimals=2)
        all_adj_matrix = np.random.choice([True, False], size=(timestamps, num_nodes, num_nodes))
        # (Apply tweaks, symmetry, set diagonal, etc.)

        # Create matrix objects
        featMatrixList, featMatrixGroup, featMatrixAnimation = displayAllMatrix(all_node_features)
        adjMatrixList, adjMatrixGroup, adjMatrixAnimation = displayAllMatrix(all_adj_matrix)

        # Setup and play feature matrix animations
        firstFeatMatrix = featMatrixList[0]
        t1FeatTitle = titleObj("Feature Matrix", firstFeatMatrix)
        t1NodeAmtText = braceMatrix("Node Amount", firstFeatMatrix, LEFT)
        t1NodeFeatText = braceMatrix("Node Features", firstFeatMatrix, DOWN)
        t1Text = VGroup(t1NodeAmtText, t1NodeFeatText)

        self.play(featMatrixAnimation[0])
        self.play(FadeIn(t1Text), FadeIn(t1FeatTitle))
        self.wait(0.5)
        
        # Show indices, then transition to the adjacency matrix
        featRowIndicesList, featColIndicesList, _ = indexMatrix(firstFeatMatrix)
        self.play(FadeOut(t1Text))
        self.play(FadeIn(VGroup(*featRowIndicesList), shift=LEFT))
        self.wait(0.5)
        
        # Position and animate the adjacency matrix.
        adjMatrixGroup.next_to(featMatrixGroup, DOWN*2.4)
        colorListofMatrices(adjMatrixList, all_adj_matrix)
        self.play(AnimationGroup(*adjMatrixAnimation, lag_ratio=0.1))
        
        # Label timestamps and add an episode title.
        featTimeBraceList, featTimeBraceGroup = labelTimestamps(featMatrixList)
        adjTimeBraceList, adjTimeBraceGroup = labelTimestamps(adjMatrixList)
        episodeTitle = titleObj("Episode Matrices", VGroup(featMatrixGroup, adjMatrixGroup))
        self.play(FadeIn(featTimeBraceGroup), FadeIn(adjTimeBraceGroup), FadeIn(episodeTitle))
        self.wait(1)
        self.play(FadeOut(featTimeBraceGroup), FadeOut(adjTimeBraceGroup))
        # End of Scene 1.
        self.wait(1)

####################################
# Scene 2: Ego Mask Selection and Application
####################################
class EgoMaskScene(MovingCameraScene):
    def construct(self):
        # (For simplicity, you might reinitialize the same matrices or load from saved state.)
        # Here we assume that the adjacency matrices and corresponding ego mask have been created.
        np.random.seed(0)
        num_nodes = 7
        timestamps = 3
        # ... (other parameters)
        all_adj_matrix = np.random.choice([True, False], size=(timestamps, num_nodes, num_nodes))
        # (Apply symmetry and set diagonal as in Scene 1)
        ego_random_idx = 1
        ego_mask = all_adj_matrix[:, ego_random_idx, :]

        # Create matrix objects for adjacency and ego mask.
        _, adjMatrixGroup, adjMatrixAnimation = displayAllMatrix(all_adj_matrix)
        egoMatrixList, egoMatrixGroup, egoMatrixAnimation = displayAllMatrix(ego_mask)

        # Show the chosen ego node.
        egoNodeIndexTitle = titleObj(f"Pick random Ego Node Index: {ego_random_idx}", adjMatrixGroup, DOWN)
        self.play(FadeIn(egoNodeIndexTitle))
        # Highlight the corresponding boxes (assume makeSurroundRectList and moveBoxesList work as before)
        adjBoxesList, _ = makeSurroundRectList([adjMatrixGroup])
        # For demonstration, we highlight the ego column.
        self.play(FadeIn(adjBoxesList[0][ego_random_idx]))
        self.wait(0.5)
        
        # Move ego mask into place.
        egoMatrixGroup.next_to(adjMatrixGroup, DOWN*2.4)
        self.play(FadeOut(egoNodeIndexTitle))
        self.play(FadeIn(egoMatrixGroup))
        # Add a title.
        egoMaskTitle = titleObj("Apply Ego Mask", egoMatrixGroup, DOWN)
        self.play(FadeIn(egoMaskTitle))
        self.wait(1)
        # End of Scene 2.
        self.wait(1)

####################################
# Scene 3: Masked Episode Matrices
####################################
class MaskedEpisodeScene(MovingCameraScene):
    def construct(self):
        # Assume you have already computed zipped matrices from previous scenes.
        # For instance, you may recreate or load:
        np.random.seed(0)
        num_nodes = 7
        timestamps = 3
        all_node_features = np.round(np.random.rand(timestamps, num_nodes, 2)*100, decimals=2)
        all_adj_matrix = np.random.choice([True, False], size=(timestamps, num_nodes, num_nodes))
        # (Adjust matrices as needed)
        ego_random_idx = 1
        ego_mask = all_adj_matrix[:, ego_random_idx, :]

        # Create zipped matrices for feature and adjacency after applying the ego mask.
        featMatrixList, featMatrixGroup, _ = displayAllMatrix(all_node_features)
        T_egoMatrixList, _, _ = displayAllMatrix(ego_mask)  # Using the ego mask as a proxy for transposed mask

        # (Assume you have a zipperMatrix function to align the matrices.)
        zippedFeatMatrixList, zippedT_egoFeatMatrixList, zipGroupFeat = zipperMatrix(featMatrixList, T_egoMatrixList)
        # Position group as needed.
        zipGroupFeat.move_to(ORIGIN)
        self.play(FadeIn(zipGroupFeat))
        
        # Now perform masking: animate the masking process.
        maskingAnimation, backgroundRect, maskedEnt, unmaskedEnt = maskMatrix(zippedT_egoFeatMatrixList, zippedFeatMatrixList, ego_mask)
        for animGroup in maskingAnimation:
            self.play(AnimationGroup(*animGroup, lag_ratio=0.25))
        
        self.wait(1)
        # Remove background rectangles and masked entries.
        self.play(FadeOut(maskedEnt), FadeOut(backgroundRect))
        self.wait(1)
        # End of Scene 3.
        self.wait(1)

####################################
# Scene 4: GCN Pass Animation
####################################
class GCNPassScene(MovingCameraScene):
    def construct(self):
        np.random.seed(0)
        num_nodes = 7
        timestamps = 3
        hidden_dim = 7
        # (Create or load the necessary matrices: e.g. maskedFeatMatrixList, maskedAdjMatrixList)
        # For demonstration, create dummy matrices for GCN output.
        nodeAmtList = [np.random.randint(1, num_nodes+1) for _ in range(timestamps)]
        gcn_output = [np.round(np.random.rand(n, hidden_dim)*100, decimals=2) for n in nodeAmtList]

        # Create matrix objects for GCN output.
        gcnMatrixList, gcnMatrixGroup, gcnMatrixAnimation = displayAllMatrix(gcn_output)
        # Position GCN output.
        gcnMatrixGroup.to_edge(DOWN)
        GCNPassTitle = titleObj("GCN Pass", gcnMatrixGroup, UP)
        self.play(FadeIn(GCNPassTitle))
        
        # Animate GCN pass (using your gcnPass function)
        GCNanimationList = gcnPass([], [], gcnMatrixAnimation, gcnMatrixGroup)  # (Pass proper matrices here)
        for animGroup in GCNanimationList:
            self.play(AnimationGroup(*animGroup, lag_ratio=0.1))
        self.wait(1)
        # End of Scene 4.
        self.wait(1)

####################################
# Scene 5: LSTM Pass and Final Output
####################################
class LSTMPassScene(MovingCameraScene):
    def construct(self):
        np.random.seed(0)
        num_nodes = 7
        timestamps = 3
        hidden_dim = 7
        output_dim = 8
        # Create dummy LSTM padded and output matrices.
        lstm_padded = np.zeros((timestamps, num_nodes, hidden_dim))
        lstm_output = np.round(np.random.rand(1, num_nodes, output_dim)*100, decimals=2)
        union_mask = np.random.choice([True, False], size=(num_nodes,))
        
        # Create matrix objects.
        PaddedMatrixList, PaddedMatrixGroup, PaddedMatrixAnimation = displayAllMatrix(lstm_padded)
        lstmOutMatrixList, lstmOutMatrixGroup, lstmOutMatrixAnimation = displayAllMatrix(lstm_output)
        lstmpassTitle = titleObj("LSTM Pass", lstmOutMatrixGroup, DOWN)
        self.play(FadeIn(lstmpassTitle))
        
        # Animate LSTM pass (using your lstmpass function)
        lstmanimation = lstmpass(PaddedMatrixList, lstmOutMatrixList, lstmOutMatrixAnimation)
        for animGroup in lstmanimation:
            self.play(AnimationGroup(*animGroup, lag_ratio=0.1))
        
        # Final output title and braces.
        finaloutputtitle = titleObj("Final Output", lstmOutMatrixGroup, UP)
        nodeAmtBraceSeen = braceMatrix("Seen Node Amount", lstmOutMatrixGroup, LEFT)
        outputdimBrace = braceMatrix("Output Dimension", lstmOutMatrixGroup, DOWN)
        finalGroup = VGroup(finaloutputtitle, nodeAmtBraceSeen, outputdimBrace)
        self.play(FadeIn(finalGroup))
        self.wait(2)
        # End of Scene 5.
        self.wait(1)
