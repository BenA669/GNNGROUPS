import pygame
import math
from pygame.math import Vector2

def closest_point_on_segment(p, a, b):
    """
    Given a point p and a segment defined by points a and b,
    return the point on the segment that is closest to p.
    """
    ab = b - a
    if ab.length_squared() == 0:
        return a
    t = (p - a).dot(ab) / ab.length_squared()
    t = max(0, min(1, t))  # Clamp t to the [0,1] range.
    return a + t * ab

def main():
    # Initialize Pygame.
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Bouncing Ball in a Spinning Hexagon")
    clock = pygame.time.Clock()

    # --- Simulation Parameters ---
    # Hexagon settings
    hex_center = Vector2(width / 2, height / 2)
    hex_radius = 200
    hex_rotation = 0             # Initial rotation angle (in radians)
    angular_velocity = 0.01      # Rotation speed (radians per frame)

    # Ball settings
    ball_radius = 10
    # Start the ball near the center (but not exactly at the center)
    ball_pos = Vector2(hex_center.x, hex_center.y - 50)
    ball_vel = Vector2(3, -2)    # Initial velocity

    # Physics constants
    gravity = 0.3                # Downward acceleration
    damping = 0.999              # Air friction (damping factor each frame)
    restitution = 0.9            # Bounciness (coefficient of restitution)
    friction_coef = 0.9          # Friction on the tangential collision component

    running = True
    while running:
        # Limit to 60 frames per second.
        dt = clock.tick(60) / 1000  # Delta time in seconds (not used here but available if needed)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Update the Hexagon ---
        hex_rotation += angular_velocity
        vertices = []
        for i in range(6):
            angle = hex_rotation + i * (2 * math.pi / 6)
            x = hex_center.x + hex_radius * math.cos(angle)
            y = hex_center.y + hex_radius * math.sin(angle)
            vertices.append(Vector2(x, y))

        # --- Update the Ball ---
        # Apply gravity and damping (air friction)
        ball_vel.y += gravity
        ball_vel *= damping
        ball_pos += ball_vel

        # --- Collision Detection and Response ---
        # Check the ball against each edge of the hexagon.
        for i in range(len(vertices)):
            a = vertices[i]
            b = vertices[(i + 1) % len(vertices)]
            cp = closest_point_on_segment(ball_pos, a, b)
            diff = ball_pos - cp
            dist = diff.length()

            if dist < ball_radius:
                # Ensure we have a valid normal direction.
                if dist == 0:
                    diff = ball_pos - hex_center
                    if diff.length() == 0:
                        diff = Vector2(1, 0)
                    dist = diff.length()
                normal = diff.normalize()
                penetration = ball_radius - dist
                # Move the ball out of the wall to prevent “sinking.”
                ball_pos += normal * penetration

                # Compute the wall’s velocity at the contact point.
                # For a rotation about hex_center, the velocity is given by:
                # v_wall = angular_velocity * (-dy, dx)
                rel_cp = cp - hex_center
                v_wall = Vector2(-rel_cp.y, rel_cp.x) * angular_velocity

                # Compute the ball’s velocity relative to the moving wall.
                v_rel = ball_vel - v_wall
                vn = v_rel.dot(normal)  # Normal component (scalar)

                # Only respond if the ball is moving into the wall.
                if vn < 0:
                    # Decompose the relative velocity.
                    v_normal = normal * vn
                    v_tangent = v_rel - v_normal
                    # Reflect the normal component and apply friction to the tangential component.
                    new_v_rel = (-restitution * v_normal) + (friction_coef * v_tangent)
                    ball_vel = new_v_rel + v_wall

        # --- Draw Everything ---
        screen.fill((30, 30, 30))  # Dark background

        # Draw the hexagon.
        hex_points = [(v.x, v.y) for v in vertices]
        pygame.draw.polygon(screen, (200, 200, 200), hex_points, 3)

        # Draw the ball.
        pygame.draw.circle(screen, (255, 100, 100), (int(ball_pos.x), int(ball_pos.y)), ball_radius)

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
