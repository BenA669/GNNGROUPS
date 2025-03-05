from manim import *
import numpy as np
import numpy.ma as ma

BUFFER = 1.1

def displayAllMatrix(matrices):
    # Loop through matrices and make Matrix object for each
    # each object and list of animations 
    # Expected first dim of matrices is timestep
    matrixList = []
    matrixAnimations = []
    for idx, matrix in enumerate(matrices):
        #Ensure the matrix is 2D. If it's 1D, convert it to a 2D array with one row.
        matrix = np.atleast_2d(matrix)

        # Make new matrix and position it next to prev matrix
        newMatrix = Matrix(matrix, h_buff=1.5).scale(0.75)
        matrixList.append(newMatrix.next_to(matrixList[idx-1], RIGHT) if idx != 0 else newMatrix)

        # Make animation
        matrixAnimations.append(FadeIn(matrixList[idx], shift=RIGHT))

    # Make group
    matrixGroup = VGroup(*matrixList)
    return matrixList, matrixGroup, matrixAnimations

def braceMatrix(text, matrix, direction):
    newBrace = Brace(matrix, direction=direction)
    newText = Text(text).scale(0.5)

    if np.allclose(direction, DOWN):
        rotate = 0
        buff = 0
    elif np.allclose(direction, UP):
        rotate = 0
        buff = 0
    elif np.allclose(direction, LEFT):
        # rotate = np.pi/2
        rotate = 0
        buff = 0.1
    elif np.allclose(direction, RIGHT):
        # rotate = -np.pi/2
        rotate = 0
        buff = 0
    else:
        print("Wrong direction format?")

    newText.next_to(newBrace, direction=direction, buff=buff).rotate(rotate)

    braceText = VGroup(newBrace, newText)
    return braceText

def colorMatrix(matrix, truth):
    truth = truth.flatten()
    for idx, ent in enumerate(matrix.get_entries()):
        if truth[idx] == True:
            ent.set_color(GREEN)
        else:
            ent.set_color(RED)

def colorListofMatrices(matrixList, truthList, wtf=False):
    for idxMat, matrix in enumerate(matrixList):
            truth = truthList[idxMat].flatten()
            for idxEnt, ent in enumerate(matrix.get_entries()):
                if wtf:
                    if(ent.get_tex_string() == "True"):
                        ent.set_color(GREEN)
                    else:
                        ent.set_color(RED)
                else:
                    if truth[idxEnt] == True:
                        ent.set_color(GREEN)
                    else:
                        ent.set_color(RED)

def indexMatrix(matrix):
    row_indices = []
    col_indices = []
    for idx, row in enumerate(matrix.get_rows()):
        y_coord = row[0].get_y()
        row_indices.append(Integer(idx).next_to(matrix, LEFT).set_y(y_coord))

    for idx, col in enumerate(matrix.get_columns()):
        x_coord = col[0].get_x()
        col_indices.append(Integer(idx).next_to(matrix, UP).set_x(x_coord))

    indicesGroup = VGroup(*row_indices, *col_indices)
    return row_indices, col_indices, indicesGroup

def indexMatrixList(MatrixList):
    row_indicesList = []
    col_indicesList = []
    indicesGroupList = []

    for idxMat, matrix in enumerate(MatrixList):
        row_indices, col_indices, indicesGroup = indexMatrix(matrix)
        row_indicesList.append(row_indices)
        col_indicesList.append(col_indices)
        indicesGroupList.append(indicesGroup)

    return row_indicesList, col_indicesList, indicesGroupList
    

def labelTimestamps(listMatrix, textT=True):
    timeBraces = []
    for idx, matrix in enumerate(listMatrix):
        if textT:
            newBrace = braceMatrix("t = {}".format(idx), matrix, DOWN)
        else:
            newBrace = braceMatrix("t = {}".format(idx), matrix, DOWN)
        timeBraces.append(newBrace)

    timeBracesGroup = Group(*timeBraces)
    return timeBraces, timeBracesGroup

def makeSurroundRect(matrix):
    boxes = []
    for idx, row in enumerate(matrix.get_rows()):
        boxes.append(SurroundingRectangle(row, buff=0.07))
    
    boxesGroup = Group(*boxes)
    return boxes, boxesGroup

def makeSurroundRectList(matrixList):
    boxesList = []
    boxesListGroups = []
    for idxMat, matrix in enumerate(matrixList):
        boxes = []
        for idxRow, row in enumerate(matrix.get_rows()):
            boxes.append(SurroundingRectangle(row))
        boxesList.append(boxes)
        boxesListGroups.append(Group(*boxes))

    return boxesList, boxesListGroups

def makeSurroundingBigMatrix(matrixList):
    '''
    Make a surrounding Rectangle on the entire matrix.
    Return Object & Group List
    '''
    bigBoxes = []
    for idxMat, matrix in enumerate(matrixList):
        bigBoxes.append(SurroundingRectangle(matrix))
    return bigBoxes

def lstmpass(lstfilledList, lstmoutputlist, lstoutputanimation):
    animationList = []
    bigBoxFilled = makeSurroundingBigMatrix(lstfilledList)

    for idx, box in enumerate(bigBoxFilled):
        animationIter = []

        if idx == 0:
            animationIter.append(Create(box))
            animationList.append(animationIter)
            continue

        animationIter.append(ReplacementTransform(bigBoxFilled[idx-1], box))
        animationList.append(animationIter)

    animationIter = []
    animationIter.append(lstoutputanimation[0])
    animationList.append(animationIter)

    animationIter = []
    animationIter.append(Uncreate(bigBoxFilled[-1]))
    animationList.append(animationIter)

    return animationList
    

def gcnPass(featMatrixList, adjMatrixList, gcnMatrixAnimation, gcnMatrixGroup):
    animationList = []
    bigBoxFeat = makeSurroundingBigMatrix(featMatrixList)
    bigBoxAdj = makeSurroundingBigMatrix(adjMatrixList)
    # GcnOutput = titleObj("GCN Embeddings", gcnMatrixGroup, DOWN)
    # Order BoxFeat -> BoxAdj -> Reveal GCN
    for idx, bigBox in enumerate(zip(bigBoxFeat, bigBoxAdj)):
        animationIter = []
        bigBoxFeat = bigBox[0]
        bigBoxAdj = bigBox[1]
        
        if idx == 0:
            # animationIter.append(FadeIn(GcnOutput))
            animationIter.append(Create(bigBoxFeat))
            animationIter.append(Create(bigBoxAdj))
            animationIter.append(gcnMatrixAnimation[idx])
            prevbigBoxFeat = bigBoxFeat
            prevbigBoxAdj = bigBoxAdj
        else:
            animationFeatMove = ReplacementTransform(prevbigBoxFeat, bigBoxFeat)
            animationAdjMove = ReplacementTransform(prevbigBoxAdj, bigBoxAdj)
            prevbigBoxFeat = bigBoxFeat
            prevbigBoxAdj = bigBoxAdj
            animationIter.append(animationFeatMove)
            animationIter.append(animationAdjMove)
            animationIter.append(gcnMatrixAnimation[idx])

        animationList.append(animationIter)
    animationIter = []
    animationIter.append(Uncreate(bigBoxFeat[-1]))
    animationIter.append(Uncreate(bigBoxAdj[-1]))
    animationList.append(animationIter)
    return animationList


        

def moveBoxesList(boxList1, boxList2, copy=True):
    animationList = []
    if len(boxList1) != len(boxList2):
        print("Mismatch box list")
        return 0
    else:
        for i in range(len(boxList1)):
            animation = ReplacementTransform(boxList1[i].copy() if copy else boxList1[i], boxList2[i])
            animationList.append(animation)
    return animationList


def makeGroupAnimationfromGroup(group):
    animationList = []
    for object in group:
        animationList.append(Create(object))

    return animationList


def titleObj(text, object, direction=UP):
    newTitleText = Text(text).next_to(object, direction)
    return newTitleText
    


def centerCameraResize(self, object, buffer=BUFFER):
    self.play(
                self.camera.frame.animate.move_to(object.get_center()).set(width=object.get_width() * buffer)
            )

def centerCamera(self, object):
    self.play(
                self.camera.frame.animate.move_to(object.get_center())
            )

def zipperMatrix(MatrixList1, MatrixList2, scale=1):
    # Return List of zippered obj (new)
    # Animation list
    zippedMatrix1 = []
    zippedMatrix2 = []

    for idx, matrices in enumerate(zip(MatrixList1, MatrixList2)):
        matrix1 = matrices[0]
        matrix2 = matrices[1]
        if idx != 0:
            zippedMatrix2.append(matrix2.copy().next_to(zippedMatrix1[idx-1], RIGHT*3).scale(scale))
        else:
            zippedMatrix2.append(matrix2.copy().scale(scale))
        zippedMatrix1.append(matrix1.copy().next_to(zippedMatrix2[idx], RIGHT).scale(scale))

    zippedGroup = Group(*zippedMatrix1, *zippedMatrix2)
    return zippedMatrix1, zippedMatrix2, zippedGroup

def maskMatrix(maskList, MatrixList, truthList, square=False):
    # Return animation group that iterates through the rows and colors matrix red
    maskBoxesList, maskBoxesListGroups = makeSurroundRectList(maskList)
    matrixBoxesList, matrixBoxesListGroups = makeSurroundRectList(MatrixList)

    # matrixBox shape = (3, 7) of rectObj

    animationList = []
    animationIteration = []
    backgroundRect = Group()
    maskedEnt = Group()
    for idNode in range(len(maskBoxesList[0])+1):
        for idMat in range(len(maskBoxesList)):
            if idNode == len(maskBoxesList[0]):
                animationIteration.append(Uncreate(maskBoxesList[idMat][idNode-1], run_time=0.25))
                animationIteration.append(Uncreate(matrixBoxesList[idMat][idNode-1],  run_time=0.25))
                continue
                
            
            if idNode == 0:
                animationIteration.append(Create(maskBoxesList[idMat][idNode], run_time=0.25))
                animationIteration.append(Create(matrixBoxesList[idMat][idNode], run_time=0.25))
            else:
                animationIteration.append(ReplacementTransform(maskBoxesList[idMat][idNode-1], maskBoxesList[idMat][idNode], run_time=0.15))
                animationIteration.append(ReplacementTransform(matrixBoxesList[idMat][idNode-1], matrixBoxesList[idMat][idNode], run_time=0.15))

            # backgroundTest = BackgroundRectangle(adjMatrixList[0].get_rows()[0], color=RED, fill_opacity=0.35)
            #     self.add(backgroundTest)
            # check ground truth
            if truthList[idMat][idNode] == False:
                # Get row information
                targetRows = MatrixList[idMat].get_rows()[idNode]
                maskedEnt.add(targetRows)
                BackRectRow = BackgroundRectangle(targetRows, color=RED, fill_opacity=0.35)
                if square:
                    targetColumns = MatrixList[idMat].get_columns()[idNode]
                    maskedEnt.add(targetColumns)
                    BackRectCol = BackgroundRectangle(targetColumns, color=RED, fill_opacity=0.35)
                    backgroundRect.add(BackRectCol)
                    animationIteration.append(Create(BackRectCol, run_time=0.15))
                backgroundRect.add(BackRectRow)
                animationIteration.append(Create(BackRectRow, run_time=0.15))
        
        animationList.append(animationIteration)
        animationIteration = []

    # animation list shape: (7, 3) obj=animation

    unmaskedEnt = Group()
    unmaskedEntRet = []
    # Loop through each matrix in MatrixList.
    for idMat, matrix in enumerate(MatrixList):
        # Get the rows (or entries) from the matrix.
        rows = matrix.get_rows()  # Assuming each row corresponds to an entry in truthList.
        # Only iterate over the rows that were processed (exclude the extra uncreate iteration).
        for idNode, row in enumerate(rows[:len(maskBoxesList[0])]):
            # If the truthList indicates this entry is NOT masked (i.e. True), add it.
            if truthList[idMat][idNode]:
                unmaskedEnt.add(row)
        unmaskedEntRet.append(unmaskedEnt)
        unmaskedEnt = Group()
                
    return animationList, backgroundRect, maskedEnt, unmaskedEntRet

def CleanMatrix(matrix, mask, self, big=False):
    matrix_pos = matrix.get_center()
    rowAmt = int(mask.sum())
    rows = matrix.get_rows()
    cols = matrix.get_columns()
    colAmt = int(len(cols))

    entries = matrix.get_entries()

    saved_rows = Group()
    # for row_id, row in enumerate(rows):
    #     if mask[row_id]:
    #         saved_rows.add(row)

    for ent_id, ent in enumerate(entries):
        if big:
            if(mask[ent_id % colAmt] and mask[ent_id//colAmt]):
                saved_rows.add(ent)
        else:
            if(mask[ent_id//colAmt]):
                saved_rows.add(ent)

    brackets = matrix.get_brackets()
    self.play(saved_rows.animate(run_time=0.25).arrange_in_grid(row=rowAmt, cols=(rowAmt if big else colAmt), buff=0.35, flow_order=("rd" if big else "rd")).move_to(matrix_pos))
    cleanedRows = saved_rows.arrange_in_grid(row=rowAmt, cols=(rowAmt if big else colAmt), buff=0.35)
    self.play(brackets.animate(run_time=0.1).stretch_to_fit_height(cleanedRows.height + 0.5))
    cleanMatrix = Group(cleanedRows)
    # cleanMatrix.add(brackets)
    for brac in matrix.get_brackets():
        cleanMatrix.add(brac)
    return cleanedRows, cleanMatrix
    

def CleanMatrixList(matrixList, maskList, self, big=False):
    cleanedRowsList = []
    cleanedMatrixList = []
    for mm in zip(matrixList, maskList):
        matrix = mm[0]
        mask = mm[1]
        cleanedRows, cleanMatrix = CleanMatrix(matrix, mask, self,big)
        cleanedRowsList.append(cleanedRows)
        cleanedMatrixList.append(cleanMatrix)
    return cleanedRowsList, cleanedMatrixList

def gcnToPad(gcnMatrixList, paddedMatrixList, egoIndexList):
    GCNboxesList, GCNboxesListGroups = makeSurroundRectList(gcnMatrixList)
    PaddedboxesList, PaddedboxesListGroups = makeSurroundRectList(paddedMatrixList)
    EGOboxesList, EGOboxesListGroups = makeSurroundRectList(egoIndexList)


    # if(ent.get_tex_string() == "True"):
    #                     ent.set_color(GREEN)
    #                 else:
    #                     ent.set_color(RED)
    lengs = [len(matList.get_rows()) for matList in gcnMatrixList]
    maxLeng = max(lengs)

    animationList = []
    animationIteration = []
    GCNStacks = []
    PaddedStacks = []
    # Get Group GCN
    # Get group Padded
    for idxMat, mat in enumerate(gcnMatrixList):
        GCNStack = []
        PaddedStack = []
        gcnmatRows = mat.get_rows()
        for idxRow, row in enumerate(gcnmatRows):
            targetIndex = int(egoIndexList[idxMat].get_entries()[idxRow].get_tex_string())
            GCNStack.append(Group(row))
            PaddedStack.append(Group(paddedMatrixList[idxMat].get_rows()[targetIndex]))
        GCNStacks.append(GCNStack)
        PaddedStacks.append(PaddedStack)

    runtime = 0.2
    stickerGCN = Group()
    def replaceGroupBwA(GroupA, GroupB, animationIter):
        # animationIter.append(Wait(1))   
        GroupAcopy = GroupA.copy()     
        animationIter.append(FadeOut(GroupB, run_time=runtime))
        animationIter.append(GroupA.animate(run_time=runtime+0.1).move_to(GroupB.get_center()))
        animationIter.append(FadeIn(GroupAcopy,run_time=0))
        return GroupAcopy

    matDone = [False for i in range(len(gcnMatrixList))]
    
    nextIterAdd = []
    
    for idxNode in range(maxLeng+1):
        delayAdd = []
        for idxMat, mat in enumerate(zip(GCNboxesList, PaddedboxesList, EGOboxesList)):
            if matDone[idxMat]:
                continue

            GCNList = mat[0]
            PaddedList = mat[1]
            EGOList = mat[2]

            # egoMaskIndex = int(egoIndexList[idxMat, idxNode].get_tex_string())                        

            if idxNode == lengs[idxMat]:
                animationIteration.append(Uncreate(GCNList[idxNode-1], run_time=runtime))
                animationIteration.append(Uncreate(EGOList[idxNode-1], run_time=runtime))
                matDone[idxMat] = True  
                continue

            if idxNode == 0:
                animationIteration.append(Create(EGOList[idxNode], run_time=runtime))
                animationIteration.append(Create(GCNList[idxNode], run_time=runtime))
                # animationIteration.append(Create(PaddedList[idxNode]))
                
                
                # animationIteration.append(Transform(PaddedStacks[idxMat][idxNode], GCNStacks[idxMat][idxNode].copy()))

                # animationIteration.append(Create(PaddedList[egoMaskIndex]))
            else:
                animationIteration.append(ReplacementTransform(EGOList[idxNode-1], EGOList[idxNode], run_time=runtime))
                animationIteration.append(ReplacementTransform(GCNList[idxNode-1], GCNList[idxNode], run_time=runtime))
                # animationIteration.append(ReplacementTransform(PaddedList[idxNode-1], PaddedList[idxNode]))
            sticker = replaceGroupBwA(GCNStacks[idxMat][idxNode], PaddedStacks[idxMat][idxNode], delayAdd)
            stickerGCN.add(sticker)
                # animationIteration.append(Transform(PaddedStacks[idxMat][idxNode], GCNStacks[idxMat][idxNode].copy()))
            
            
        animationIteration.extend(delayAdd)
        animationList.append(animationIteration)
        animationIteration = []

    # animationIteration.append(Uncreate(GCNList[-1]))
    # animationIteration.append(Uncreate(PaddedList[-1]))
    # animationIteration.append(Uncreate(EGOList[-1]))
    # animationList.append(animationIteration)

    return animationList, stickerGCN
    
    

class Big(MovingCameraScene):
    def construct(self):
        np.random.seed(0)
        # Num nodes = 8
        # Num Timestamps = 4
        # Output dim = ?

        num_nodes = 7
        timestamps = 3
        node_features = 2
        hidden_dim = 7
        output_dim = 8

        ego_random_idx = 1


        # Shape = (T, Num_Nodes, Features)
        all_node_features = np.round(np.random.rand(timestamps, num_nodes, node_features)*100, decimals=2)

        # Shape = (T, Num Nodes, Num Nodes)
        all_adj_matrix = np.random.choice([True, False], size=(timestamps, num_nodes, num_nodes))

        # tweak for *aesthetics*
        all_adj_matrix[2][1][-1] = False

        # Force symmetry along the diagonal:
        all_adj_matrix = np.triu(all_adj_matrix) | np.triu(all_adj_matrix).transpose(0, 2, 1)
        # Set the diagonal of each matrix to True:
        all_adj_matrix[:, np.arange(num_nodes), np.arange(num_nodes)] = True

        # Shape = (T, Num Nodes)
        ego_mask = all_adj_matrix[:, ego_random_idx, :]
        ego_mask_T = np.where(np.resize(ego_mask, (timestamps, num_nodes, 1)), "T", "F")
        ego_mask_I = [list(np.where(row)[0]) for row in ego_mask]
        ego_mask_I_T = []
        for idx, mat in enumerate(ego_mask_I):
            t_mat = []
            for ent in mat:
                t_mat.append([ent])
            ego_mask_I_T.append(t_mat)

        union_mask = np.any(ego_mask, axis=0)
        union_mask_expanded = np.expand_dims(union_mask, axis=0)
        # Reshape to mimic the transposed structure of ego_mask_T.
        # Here we use shape (1, num_nodes, 1) since we collapsed the T dimension.
        egomask_union_T = np.where(union_mask.reshape(1, num_nodes, 1), "T", "F")

        # Shape = (T, X, 2)
        masked_node_features = [subMat[ego_mask[idMat]] for idMat, subMat in enumerate(all_node_features)]

        # Shape = (T, X, X)
        masked_adj_matrix = [subMat[ego_mask[idMat]][:, ego_mask[idMat]] for idMat, subMat in enumerate(all_adj_matrix)]

        # LSTM Padded = (T, Num Nodes, Hidden Dim)
        lstm_padded = np.zeros(shape=(timestamps, num_nodes, hidden_dim))

        lstm_output = np.round(np.random.rand(1, num_nodes, output_dim)*100, decimals=2)

        lstm_output_filtered = lstm_output[:, union_mask, :]
        

        featMatrixList, featMatrixGroup, featMatrixAnimation = displayAllMatrix(all_node_features)
        adjMatrixList, adjMatrixGroup, adjMatrixAnimation = displayAllMatrix(all_adj_matrix)
        egoMatrixList, egoMatrixGroup, egoMatrixAnimation = displayAllMatrix(ego_mask)
        T_egoMatrixList, T_egoMatrixGroup, T_egoMatrixAnimation = displayAllMatrix(ego_mask_T)
        I_egoMatrixList, I_egoMatrixGroup, I_egoMatrixAnimation = displayAllMatrix(ego_mask_I)
        T_I_egoMatrixList, T_I_egoMatrixGroup, T_I_egoMatrixAnimation = displayAllMatrix(ego_mask_I_T)
        maskedFeatMatrixList, maskedFeatMatrixGroup, maskedFeatMatrixAnimation = displayAllMatrix(masked_node_features)
        maskedAdjMatrixList, maskedAdjMatrixGroup, maskedAdjMatrixAnimation = displayAllMatrix(masked_adj_matrix)
        PaddedMatrixList, PaddedMatrixGroup, PaddedMatrixAnimation = displayAllMatrix(lstm_padded)
        lstmOutMatrixList, lstmOutMatrixGroup, lstmOutMatrixAnimation = displayAllMatrix(lstm_output)
        egoUnionMatrixList, egoUnionMatrixGroup, egoUnionMatrixAnimation = displayAllMatrix(egomask_union_T)
        lstmoutputFilteredMatrixList, lstmoutputFilteredMatrixGroup, lstmoutputFilteredMatrixAnimation = displayAllMatrix(lstm_output_filtered)
        
        firstFeatMatrix = featMatrixList[0]
        t1FeatTitle = titleObj("Feature Matrix", firstFeatMatrix)
        t1NodeAmtText = braceMatrix("Node Amount", firstFeatMatrix, LEFT)
        t1NodeFeatText = braceMatrix("Node Features", firstFeatMatrix, DOWN)
        t1Text = Group(t1NodeAmtText, t1NodeFeatText)

        # Flash t1 Feat Text
        self.play(featMatrixAnimation[0])
        self.play(FadeIn(t1Text), FadeIn(t1FeatTitle))

        self.wait(0.5)
        
        # Flash Feat indices
        featRowIndicesList, featColIndicesList, featIndicesGroup = indexMatrix(firstFeatMatrix)

        self.play(FadeOut(t1Text))
        self.play(FadeIn(Group(*featRowIndicesList), shift=LEFT))

        self.wait(0.5)

        # Adj matrix proccessing
        adjMatrixGroup.next_to(featMatrixGroup, DOWN*2.4) # Position adjMatrixGroup
        colorListofMatrices(adjMatrixList, all_adj_matrix) # Color adjMatrix
        colorListofMatrices(egoMatrixList, ego_mask) # Color egoMatrix
        colorListofMatrices(T_egoMatrixList, ego_mask)
        colorListofMatrices(egoUnionMatrixList, union_mask_expanded)
        colorListofMatrices(maskedAdjMatrixList, ego_mask, wtf=True) 

        # t1 sample proccessing
        adjSamplet1 = adjMatrixList[0].copy()
        adjSamplet1.next_to(firstFeatMatrix, RIGHT*20)
        t1AdjTitle = titleObj("Adjacency Matrix", adjSamplet1)
        t1AdjNodeAmtText1 = braceMatrix("Node \nAmount", adjSamplet1, LEFT)
        t1AdjNodeAmtText2 = braceMatrix("Node Amount", adjSamplet1, DOWN)
        t1AdjText = Group(t1AdjNodeAmtText1, t1AdjNodeAmtText2)

        # Flash t1 Adj
        centerCamera(self, adjSamplet1)
        self.play(FadeIn(adjSamplet1, shift=RIGHT))
        self.play(FadeIn(t1AdjText), FadeIn(t1AdjTitle))

        self.wait(0.5)

        # Flash adj indices
        adjRowIndicesList, adjColIndicesList, adjIndicesGroup = indexMatrix(adjSamplet1)
        self.play(FadeOut(t1AdjText), t1AdjTitle.animate.next_to(Group(*adjColIndicesList), UP))
        self.play(FadeIn(Group(*adjRowIndicesList), shift=LEFT), FadeIn(Group(*adjColIndicesList), shift=UP))

        self.wait(0.5)

        # Cleanup & Resize

        objToClean = Group(adjIndicesGroup, t1AdjTitle, *featRowIndicesList, t1FeatTitle)
        featAdjGroup = Group(featMatrixGroup, adjMatrixGroup)
        centerCameraResize(self, featAdjGroup)

        self.wait(0.7)

        
        self.play(FadeOut(objToClean), ReplacementTransform(adjSamplet1, adjMatrixList[0]))

        # Move adj matrix
        

        # Show timestamp
        self.play(AnimationGroup(*featMatrixAnimation[1:], lag_ratio=0.1), 
                  AnimationGroup(*adjMatrixAnimation[1:], lag_ratio=0.1))
        
        # flash t = 0, t = 1 ... & and show Episode title
        featTimeBraceList, featTimeBraceGroup = labelTimestamps(featMatrixList)
        adjTimeBraceList, adjTimeBraceGroup = labelTimestamps(adjMatrixList)
        episodeTitle = titleObj("Episode Matrices", featAdjGroup)
        self.play(FadeIn(featTimeBraceGroup), FadeIn(adjTimeBraceGroup), FadeIn(episodeTitle))

        self.wait(1)
        
        self.play(FadeOut(featTimeBraceGroup), FadeOut(adjTimeBraceGroup))

        # Highlight chosen ego node
        egoNodeIndexTitle = titleObj("Pick random Ego Node Index: {}".format(ego_random_idx), adjMatrixGroup, DOWN)
        self.play(FadeIn(egoNodeIndexTitle))

        adjBoxesList, adjBoxesGroup = makeSurroundRectList(adjMatrixList)
        ego2boxesList = [sublist[ego_random_idx] for sublist in adjBoxesList]
        ego2Boxes = Group(*ego2boxesList)
        self.play(AnimationGroup(*makeGroupAnimationfromGroup(ego2Boxes), lag_ratio=0.1))

        # Show subset ego mask
        # Setup ego mask positions
        egoMatrixGroup.next_to(adjMatrixGroup, DOWN*2.4)
        startMatrix = [matrix.get_rows()[ego_random_idx] for matrix in adjMatrixList]
        endMatrix = [matrix.get_rows()[0] for matrix in egoMatrixList]
        self.play(AnimationGroup(*moveBoxesList(startMatrix, endMatrix)), FadeOut(egoNodeIndexTitle))
        self.add(egoMatrixGroup)

        self.wait(0.5 )
        self.play(FadeOut(ego2Boxes))


        # Add title
        egoMaskTitle = titleObj("Apply Ego Mask", egoMatrixGroup, DOWN)
        self.play(FadeIn(egoMaskTitle))    

        # Setup zipped positions  
        zippedFeatMatrixList, zippedT_egoFeatMatrixList, zipGroupFeat = zipperMatrix(featMatrixList, T_egoMatrixList)
        zippedAdjMatrixList, zippedT_egoAdjMatrixList, zipGroupAdj = zipperMatrix(adjMatrixList, T_egoMatrixList)
        zipGroupFeat.match_y(featMatrixGroup).match_x(featAdjGroup)
        zipGroupAdj.match_y(adjMatrixGroup).match_x(featAdjGroup)

        # Move to zipped positions
        # self.remove(featMatrixGroup)
        centerCameraResize(self, zipGroupAdj)
        self.play(AnimationGroup(*moveBoxesList(featMatrixList, zippedFeatMatrixList, copy=False), lag_ratio=0.1), 
                  AnimationGroup(*moveBoxesList(egoMatrixList, zippedT_egoFeatMatrixList), lag_ratio=0.5))
        self.play(AnimationGroup(*moveBoxesList(adjMatrixList, zippedAdjMatrixList, copy=False), lag_ratio=0.1),
                  AnimationGroup(*moveBoxesList(egoMatrixList, zippedT_egoAdjMatrixList), lag_ratio=0.5))

        # Perform masking
        maskingAnimationAdj, backRectAdj, maskedEntAdj, unmaskedEntAdj = maskMatrix(zippedT_egoAdjMatrixList, zippedAdjMatrixList, ego_mask, square=True)
        maskingAnimationFeat, backRectFeat, maskedEntFeat, unmaskedEntFeat = maskMatrix(zippedT_egoFeatMatrixList, zippedFeatMatrixList, ego_mask)

        # Highlight Rect moving
        for idx, animationIter in enumerate(zip(maskingAnimationAdj, maskingAnimationFeat)):
            animationIterAdj = animationIter[0]
            animationIterFeat = animationIter[1]
            self.play(AnimationGroup(*animationIterFeat, *animationIterAdj, lag_ratio=0.25))

        self.wait(1)

        # Remove masked entires, transposed mask and background rect
        self.play(FadeOut(maskedEntAdj), FadeOut(maskedEntFeat), 
                  FadeOut(backRectAdj), FadeOut(backRectFeat), 
                  FadeOut(Group(*zippedT_egoAdjMatrixList)), 
                  FadeOut(Group(*zippedT_egoFeatMatrixList)))
        
        self.wait(1)
        self.play(FadeOut(episodeTitle), FadeOut(egoMaskTitle), FadeOut(egoMatrixGroup))
        maskedEpisodeTitle = titleObj("Masked Episode Matrices", featAdjGroup)
        # Remove masked rows and columns
        maskedAdjMatrixGroup.move_to(adjMatrixGroup)
        maskedFeatMatrixGroup.move_to(featMatrixGroup)
        cleanedEntMatrixList, cleanedMatrixListFeat = CleanMatrixList(zippedFeatMatrixList, ego_mask, self)
        cleanedEntAdjList, cleanedMatrixListAdj = CleanMatrixList(zippedAdjMatrixList, ego_mask, self, big=True)

        # More cleanup , convert to maskedLists
        self.play(AnimationGroup(FadeIn(maskedEpisodeTitle), *moveBoxesList(cleanedMatrixListAdj, maskedAdjMatrixList, copy=False), lag_ratio=0.1),
                  AnimationGroup(*moveBoxesList(cleanedMatrixListFeat, maskedFeatMatrixList, copy=False), lag_ratio=0.1))
        
        
        self.wait(1)
        # GCN Pass title
        GCNPassTitle = titleObj("GCN Pass", maskedAdjMatrixGroup, DOWN)
        self.play(FadeIn(GCNPassTitle))

        # Shape(T)
        nodeAmtList = [int(sublist.sum()) for sublist in ego_mask]
        
        # Create GCN Output
        # Shape(T, X, Hidden Dim)
        gcn_output = [np.round(np.random.rand(node_amt, hidden_dim)*100, decimals=2) for node_amt in nodeAmtList]

        # Prepare LSTM padded filled
        # LSTM Padded filled
        lstm_padded_filled = []
        for idMat, mat in enumerate(lstm_padded):
            mat[ego_mask_I[idMat]] = gcn_output[idMat]
            lstm_padded_filled.append(mat)
            
        # Prepare GCN Output Matrices
        gcnMatrixList, gcnMatrixGroup, gcnMatrixAnimation = displayAllMatrix(gcn_output)
        lstmfillMatrixList, lstmfillMatrixGroup, lstmfillgcnMatrixAnimation = displayAllMatrix(lstm_padded_filled)
        gcnMatrixGroup.next_to(GCNPassTitle, DOWN)
        
        # Brace Hidden Dim
        hiddenDimBrace = braceMatrix("Hidden Dim", gcnMatrixList[0], DOWN)
        # Brace Node Amt at time t
        nodeAmtX = braceMatrix("Node Amount\n at t", gcnMatrixList[0], LEFT)

        # GCN Embeddings
        gcnEmbedBrace = braceMatrix("GCN Embeddings", gcnMatrixGroup, DOWN)
        #GCN Pass
        GCNrow_indices, GCNcol_indices, _ = indexMatrix(gcnMatrixList[0])
        GCNanimationList = gcnPass(maskedFeatMatrixList, maskedAdjMatrixList, gcnMatrixAnimation, gcnMatrixGroup)
        for animationIter in GCNanimationList:
            self.play(AnimationGroup(*animationIter, lag_ratio=0.1))

        # GCN YAP
        GCNyap = Group()
        # GCNyap.add(gcnEmbedBrace)
        GCNyap.add(hiddenDimBrace)
        GCNyap.add(nodeAmtX)
        self.play(AnimationGroup(FadeIn(GCNyap), FadeIn(Group(*GCNcol_indices), shift=UP), lag_ratio=0.1))
        
        self.wait(2)

        # NOW CLEAN and move GCN embed UP
        self.play(FadeOut(GCNyap), FadeOut(Group(*GCNcol_indices)), GCNPassTitle.animate.next_to(maskedFeatMatrixGroup, UP),
                  FadeOut(maskedFeatMatrixGroup), FadeOut(maskedAdjMatrixGroup),
                  FadeOut(maskedEpisodeTitle), 
                  gcnMatrixGroup.animate.move_to(maskedFeatMatrixGroup.get_center()))
        
        # Bring back ego matrix group
        egoMaskBACK = titleObj("Original Ego Mask", gcnMatrixGroup, DOWN*1.5)
        egoMatrixGroup.next_to(egoMaskBACK, DOWN)
        self.play(FadeIn(egoMatrixGroup), FadeIn(egoMaskBACK))

        self.wait(0.75)
        self.play(FadeOut(egoMaskBACK))
        # Index show
        # indexMatrix(egoMatrixList)
        EGOrow_indicesList, EGOcol_indicesList, EGOindicesGroupList = indexMatrixList(egoMatrixList)

        maskedinteger = Group()
        EgoColGroup = Group()
        for idx_group, i_group in enumerate(EGOcol_indicesList):
            self.play(AnimationGroup(FadeIn(Group(*i_group), shift=UP, run_time=0.25), lag_ratio=0.1))

            # Get ego mask brackets
            egoMaskBrackets = egoMatrixList[idx_group].get_brackets().copy()
            maskedinteger.add(egoMaskBrackets[0])

            # COLORING
            colorChangeAnimation = []
            for idx_i, integer in enumerate(i_group):
                if ego_mask[idx_group][idx_i]:
                    colorChangeAnimation.append(integer.animate(run_time=0.2).set_color(GREEN))
                    maskedinteger.add(integer)
                else:
                    EgoColGroup.add(integer)
                    colorChangeAnimation.append(integer.animate(run_time=0.2).set_color(RED))

            maskedinteger.add(egoMaskBrackets[1])
            self.play(AnimationGroup(*colorChangeAnimation, lag_ratio=0.1))
        self.wait(0.5)

        # Make it index
        # I_egoMatrixList, I_egoMatrixGroup, I_egoMatrixAnimation
        I_egoTitle = titleObj("Ego Indices", egoMatrixGroup, DOWN*1.5)
        I_egoMatrixGroup.next_to(I_egoTitle, DOWN) # Setup I ego
        
        I_egoEntries = Group()
        # Get Ego I all entries
        for mat in I_egoMatrixList:
            matEntries = mat.get_entries()
            matbrackets = mat.get_brackets()
            I_egoEntries.add(matbrackets[0])
            for entry in matEntries:
                I_egoEntries.add(entry)
            I_egoEntries.add(matbrackets[1])

        # Show I mask
        self.play(ReplacementTransform(maskedinteger, I_egoEntries), FadeIn(I_egoTitle))
        # self.play(ReplacementTransform(maskedinteger, I_egoMatrixGroup))

        # self.play(AnimationGroup(*I_egoMatrixAnimation, FadeIn(I_egoTitle), lag_ratio=0.1))


        # self.wait(1)

        # Unload Old EGO move I EGO up
        self.play(FadeOut(EgoColGroup), FadeOut(egoMatrixGroup),
                  I_egoTitle.animate.next_to(egoMatrixGroup, UP),
                  I_egoMatrixGroup.animate.move_to(egoMatrixGroup.get_center()))

        self.wait(0.6) # Flash yap 

        # Setup Transpose I_Ego and zipper
        zippedGCNMatrixList, zippedT_I_egoMatrixList, zippedGroupGCNI_EGO_T = zipperMatrix(gcnMatrixList, T_I_egoMatrixList)
        zippedGroupGCNI_EGO_T.move_to(gcnMatrixGroup.get_center())

        
        # ZIPPER
        self.play(AnimationGroup(FadeOut(I_egoTitle), *moveBoxesList(gcnMatrixList, zippedGCNMatrixList, copy=False), lag_ratio=0.1),
                AnimationGroup(*moveBoxesList(I_egoMatrixList, zippedT_I_egoMatrixList, copy=False), lag_ratio=0.25))        

        # Padded matrix
        paddedTitle = titleObj("Apply Padding", zippedGroupGCNI_EGO_T, DOWN*1.5)
        PaddedMatrixGroup.next_to(paddedTitle, DOWN)

        self.play(AnimationGroup(FadeIn(paddedTitle), *PaddedMatrixAnimation, lag_ratio=0.1))
        # PaddedMatrixList, PaddedMatrixGroup, PaddedMatrixAnimation

        # Yap sesh
        paddedYapping = Group()
        paddedBraceLeft = braceMatrix("Total\n Node Amount", PaddedMatrixList[0], LEFT)
        paddedBraceDown = braceMatrix("Total Node Amount", PaddedMatrixList[0], DOWN)
        paddedYapping.add(paddedBraceDown).add(paddedBraceLeft)

        self.play(FadeIn(paddedYapping))

        self.wait(0.6) # Flash yap 

        self.play(FadeOut(paddedYapping))


        Paddedrow_indices, Paddedcol_indices, _ = indexMatrix(PaddedMatrixList[0])
        FadeIn(Group(*Paddedrow_indices), shift=LEFT)

        # Yap over
        zippedcentersaved = zippedGroupGCNI_EGO_T.get_center()

        # Apply padding time
        gcnToPadAnimationList, stickerGCN = gcnToPad(zippedGCNMatrixList, PaddedMatrixList, zippedT_I_egoMatrixList)
        for animationIteration in gcnToPadAnimationList:
            self.play(AnimationGroup(*animationIteration, lag_ratio=0.2))

        # Replace padded with filled
        # lstmfillMatrixList, lstmfillMatrixGroup, lstmfillgcnMatrixAnimation
        lstmfillMatrixGroup.move_to(PaddedMatrixGroup.get_center())
        lstmfillEnt = Group()
        movedGCNembed = Group()
        rowCounter = 0
        for idxMat, mat in enumerate(lstmfillMatrixList):
            for idxRow, row in enumerate(mat.get_rows()):
                lstmfillEnt.add(row)
                # print(ego_mask[idxMat][idxRow])
                if (ego_mask[idxMat][idxRow]):
                    movedGCNembed.add(zippedGCNMatrixList[idxMat].get_rows()[rowCounter])
                    rowCounter += 1
                else:
                    movedGCNembed.add(PaddedMatrixList[idxMat].get_rows()[idxRow])
            # print(rowCounter)
            rowCounter = 0

        bracketshelp = Group()
        for mat in PaddedMatrixList:
            bracketshelp.add(mat.get_brackets())
        
        backetswhatthe = Group()
        for mat in lstmfillMatrixList:
            backetswhatthe.add(mat.get_brackets())

        self.play(ReplacementTransform(movedGCNembed, lstmfillEnt, run_time=0.1), ReplacementTransform(bracketshelp, backetswhatthe, run_time=0.1))

        # Fade out GCN stuff
        thingsToFadeOut = Group()
        for mat in zippedT_I_egoMatrixList:
            thingsToFadeOut.add(mat)
        for mat in zippedGCNMatrixList:
            thingsToFadeOut.add(mat.get_brackets())
        


        self.play(FadeOut(GCNPassTitle), FadeOut(stickerGCN), FadeOut(thingsToFadeOut), FadeOut(paddedTitle),
                  lstmfillMatrixGroup.animate.move_to(zippedcentersaved))
        # paddedTitle.animate.next_to(zippedGroupGCNI_EGO_T, UP*1.5)
        self.wait(1)

        # Title lstm pass below lstm filled
        # show new matrix (NodeAmt, output_dim) title lstm output
        # Union mask all mask over timestep

        # LSTM PAss
        # lstmOutMatrixList, lstmOutMatrixGroup, lstmOutMatrixAnimation
        lstmpassTitle = titleObj("LSTM Pass", lstmfillMatrixGroup, DOWN*1.5)
        lstmOutputTitle = titleObj("LSTM Output", lstmfillMatrixGroup, DOWN*1.5)
        lstmOutMatrixGroup.next_to(lstmpassTitle, DOWN)
        # lstmOutputTitle.next_to(lstmOutMatrixGroup, DOWN)
        
        self.play(FadeIn(lstmpassTitle))

        lstmanimation = lstmpass(lstmfillMatrixList, lstmOutMatrixList, lstmOutMatrixAnimation)
        for animationiterlstm in lstmanimation:
            self.play(AnimationGroup(*animationiterlstm, lag_ratio=0.1))

        # self.play(FadeIn(lstmOutputTitle))

        # MOVE AND CLEAN

        self.play(FadeOut(lstmfillMatrixGroup),
                  lstmOutMatrixGroup.animate.move_to(lstmfillMatrixGroup.get_center()),
                  lstmpassTitle.animate.next_to(lstmfillMatrixGroup, UP))
        

        # Union ego mask
        unionEgoMaskTitle = titleObj("Apply Union to Ego Mask",lstmOutMatrixGroup , DOWN*1.5)
        filterLstmTitle = titleObj("Filter LSTM Output",lstmOutMatrixGroup , DOWN*1.5)
        T_egoMatrixGroup.next_to(filterLstmTitle, DOWN)
        unionEgoMaskTitle.next_to(T_egoMatrixGroup, DOWN)
        egoUnionMatrixGroup.move_to(T_egoMatrixGroup.get_center())
        
        self.play(FadeIn(filterLstmTitle)) # Filter LSTM output title
        self.play(AnimationGroup(*T_egoMatrixAnimation, lag_ratio=0.1)) # show transposed ego mask
        self.play(FadeIn(unionEgoMaskTitle)) # Show title apply union to ego mask
        self.play(ReplacementTransform(T_egoMatrixGroup, egoUnionMatrixGroup)) # Replacement transform to union ego mask
        self.play(FadeOut(unionEgoMaskTitle)) # Take out apply title
        self.play(egoUnionMatrixGroup.animate.next_to(lstmOutMatrixGroup, LEFT))
        # Get transposed ego masks and put them in a line
        # T_egoMatrixList, T_egoMatrixGroup, T_egoMatrixAnimation
        # egoUnionMatrixList, egoUnionMatrixGroup, egoUnionMatrixAnimation

        # Masking time
        lstmFilteranimationList, lstmFilterbackgroundRect, lstmFiltermaskedEnt, lstmFilterunmaskedEntRet = maskMatrix(egoUnionMatrixList, lstmOutMatrixList, union_mask_expanded)

        for animationIter in lstmFilteranimationList:
            self.play(AnimationGroup(*animationIter, lag_ratio=0.1))

        # Take out masked rows
        # lstmoutputFilteredMatrixList, lstmoutputFilteredMatrixGroup, lstmoutputFilteredMatrixAnimation
        self.play(FadeOut(lstmFiltermaskedEnt), FadeOut(egoUnionMatrixGroup), FadeOut(lstmpassTitle),
                  FadeOut(filterLstmTitle), FadeOut(lstmFilterbackgroundRect))
        
        cleanedLstmOutMatrixList, cleanedLstmMatrixListEnt = CleanMatrixList(lstmOutMatrixList, union_mask_expanded, self)
        
        lstmoutputFilteredMatrixGroup.move_to(lstmOutMatrixGroup.get_center())

        self.play(AnimationGroup(*moveBoxesList(cleanedLstmMatrixListEnt, lstmoutputFilteredMatrixList, copy=False), lag_ratio=0.1))
        finaloutputtitle = titleObj("Final Output", lstmoutputFilteredMatrixGroup , UP)
        nodeAmtBraceSeen = braceMatrix("Seen Node Amount", lstmoutputFilteredMatrixGroup, LEFT)
        outputdimBrace = braceMatrix("Output Dimension", lstmoutputFilteredMatrixGroup, DOWN)
        yapObjFinalOutput = Group()
        yapObjFinalOutput.add(nodeAmtBraceSeen)
        yapObjFinalOutput.add(outputdimBrace)
        yapObjFinalOutput.add(finaloutputtitle)
        self.play(FadeIn(yapObjFinalOutput))
        
        
        self.wait(5)