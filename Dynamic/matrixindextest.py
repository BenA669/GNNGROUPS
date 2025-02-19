from manim import *

class SubmatrixTransformScene(Scene):
    def construct(self):
        # Create the original matrix
        matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Highlight the submatrix to extract
        rect = SurroundingRectangle(matrix.get_entries_by_index([[0, 0], [1, 0], [2, 0]]))
        self.add(matrix, rect)
        
        # Create the submatrix
        submatrix = Matrix([[1, 2, 3], [4, 5, 6]])
        
        # Animate the transformation
        self.play(Transform(matrix, submatrix,  
                            path_arc=np.pi/2,  # Optional: Add a smooth curve
                            run_time=2))
        
        # Optional: Add visual feedback
        self.wait(1)
        self.play(FadeOut(rect))
        

class MatrixToSubmatrix(Scene):
    def construct(self):
        # Define the original matrix
        original_matrix = Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Define the submatrix (e.g., removing the first row and first column)
        submatrix = Matrix([
            [5, 6],
            [8, 9]
        ])

        # Position the matrices
        original_matrix.to_edge(UP)
        submatrix.move_to(original_matrix.get_center())

        # Display the original matrix
        self.play(Write(original_matrix))
        self.wait(1)

        # Transform the original matrix to the submatrix
        self.play(
            Transform(original_matrix.get_rows()[1:], submatrix.get_rows()),
            FadeOut(original_matrix.get_rows()[0]),
            FadeOut(original_matrix.get_columns()[0])
        )

        self.wait(2)

# To run the scene, use the following command in the terminal:
# manim -pql script_name.py MatrixToSubmatrix