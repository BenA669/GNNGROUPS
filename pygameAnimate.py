import pygame
from pygame_screen_recorder import pygame_screen_recorder as pgr

def animate(positions_t_n_xy, n_hop_adjacency_t_h_n_n, num_timesteps, num_nodes, scale=50, ego_idx=None, nhops=2):
    pygame.init()
    screen = pygame.display.set_mode((420, 420))
    screen_width, screen_height = screen.get_size()
    recrdr = pgr("./animation.mp4")
    center_x = screen_width // 2
    center_y = screen_height // 2
    clock = pygame.time.Clock()
    running = True
    t = 0
    once = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # Clear the screen with black

        for n in range(num_nodes):
            x, y = positions_t_n_xy[t, n, :]
            

            if  ego_idx is None:
                indicies = n_hop_adjacency_t_h_n_n[t, 0, n].nonzero().squeeze()
                pygame.draw.circle(screen, (255, 255, 255), (int(x*scale)+center_x, int(y*scale)+center_y), 5)

            else:
                if n == ego_idx:
                    indicies = n_hop_adjacency_t_h_n_n[t, 0, n].nonzero().squeeze()
                    pygame.draw.circle(screen, (50, 205, 50), (int(x*scale)+center_x, int(y*scale)+center_y), 5)

                elif n in n_hop_adjacency_t_h_n_n[t, :nhops-1, ego_idx].nonzero().squeeze():
                    indicies = n_hop_adjacency_t_h_n_n[t, 0, n].nonzero().squeeze()
                    pygame.draw.circle(screen, (255, 255, 255), (int(x*scale)+center_x, int(y*scale)+center_y), 5)

                else:
                    pygame.draw.circle(screen, (255, 255, 255), (int(x*scale)+center_x, int(y*scale)+center_y), 5)
                    continue
            
            if indicies.ndim == 0:
                continue
            for inx in indicies:
                if inx != n:
                    x2, y2 = positions_t_n_xy[t, inx, :]
                    pygame.draw.line(screen, (255, 255, 255), 
                                     (int(x*scale)+center_x, int(y*scale)+center_y), 
                                     (int(x2*scale)+center_x, int(y2*scale)+center_y), 1)


        if (t + 1) % num_timesteps == 0 and not once:
            once = True
            print("Saved")
            recrdr.save()
            pygame.quit()
            break
             
        t = (t + 1) % num_timesteps
        if once == False:
            recrdr.click(screen)
        pygame.display.flip()  # Update the display
        clock.tick(20) 
    pygame.quit()