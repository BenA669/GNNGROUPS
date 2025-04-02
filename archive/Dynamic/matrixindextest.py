from manim import *

class TransformFromCopyExample(Scene):
    def construct(self):
        # Create objects A and B
        A = Square().set_color(RED).shift(LEFT * 3)
        B = Circle().set_color(BLUE).shift(RIGHT * 3)

        # Display both
        self.add(A, B)

        # Transform B into A while moving B to A's position
        self.play(Transform(B, A.copy().move_to(B)))

        # Wait before ending
        self.wait()