import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame_screen_recorder import pygame_screen_recorder as pgr
import subprocess, numpy as np, pygame
import datetime

class FFmpegRecorder:
    def __init__(self, path, size, fps=30, crf=18, preset="slow"):
        w, h = size
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}",
                "-r", str(fps),
                "-i", "-",            # read frames from stdin
                "-an",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", preset,
                "-crf", str(crf),
                path
            ],
            stdin=subprocess.PIPE
        )
        self.size = (w, h)

    def write(self, surface: pygame.Surface):
        # Pygame gives [W,H,3]; ffmpeg expects [H,W,3], row-major
        frame = pygame.surfarray.array3d(surface)          # [W,H,3]
        frame = np.rot90(frame, 3).swapaxes(0, 1).copy()   # -> [H,W,3], contiguous
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        try:
            self.proc.stdin.close()
        finally:
            self.proc.wait()

def animatev2(positions_t_n_xy, 
              adjacency_t_n_n,
              anchor_indices_n=None,
              scale=50,
              screen_size=1080,
              boundary=3):
            #   boundary=dataset_cfg['boundary']):
    
    scale = (screen_size / (boundary * 2)) - 20

    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    screen_width, screen_height = screen.get_size()
    center_x = screen_width // 2
    center_y = screen_height // 2

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_name = f"./Saved_Models_Datasets/animations/animation_{timestamp}.mp4"
    recrdr = FFmpegRecorder(file_name, (screen_width, screen_height), fps=30, crf=8, preset="slow")

    clock = pygame.time.Clock()
    running = True
    t = 0
    once = False
    draw_queue_n_xyc = []
    colors = (
        (255, 0, 255), # Pink
        (0, 255, 154), # Green
        (255, 255, 255) # White
    )
    color = colors[2]

    num_timesteps, num_nodes, _ = positions_t_n_xy.shape

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # Clear the screen with black

        for n in range(num_nodes):
            x, y = positions_t_n_xy[t, n, :]
            indicies = adjacency_t_n_n[t, n].nonzero().squeeze()

            if n in anchor_indices_n:
                color = colors[1]
            else:
                color = colors[2]
            if indicies.ndim == 0:
                draw_queue_n_xyc.append((int(x*scale)+center_x, int(y*scale)+center_y, color))
                continue
            for inx in indicies:
                if inx != n:
                    x2, y2 = positions_t_n_xy[t, inx.item(), :]
                    pygame.draw.line(screen, (255, 255, 255), 
                                     (int(x*scale)+center_x, int(y*scale)+center_y), 
                                     (int(x2*scale)+center_x, int(y2*scale)+center_y), 1)
            # pygame.draw.circle(screen, color, (int(x*scale)+center_x, int(y*scale)+center_y), 5)
            draw_queue_n_xyc.append((int(x*scale)+center_x, int(y*scale)+center_y, color))
        
        for xyc in draw_queue_n_xyc:
            pygame.draw.circle(screen, xyc[2], (xyc[0], xyc[1]), 5)
        draw_queue_n_xyc = list()


        if (t + 1) % num_timesteps == 0 and not once:
            once = True
            print("Saved")
            print(f"saving to: {file_name}")
            recrdr.close()
            pygame.quit()
            break
             
        t = (t + 1) % num_timesteps
        if once == False:
            # recrdr.click(screen)
            recrdr.write(screen)
        pygame.display.flip()  # Update the display
        clock.tick(20) 
    pygame.quit()

def init_animate():
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    screen_width, screen_height = screen.get_size()
    # recrdr = pgr("./animation.mp4")
    recrdr = FFmpegRecorder("animation.mp4", (screen_width, screen_height), fps=30, crf=8, preset="slow")
    center_x = screen_width // 2
    center_y = screen_height // 2
    clock = pygame.time.Clock()
    t = 0
    draw_queue_n_xyc = []
    colors = (
        (255, 0, 255), # Pink
        (0, 255, 154), # Green
        (255, 255, 255), # White
        (0, 0, 255), # Blue
        (255, 0, 0), # Red
    )
    once = False
    return screen, colors, t, center_x, center_y, clock, recrdr, draw_queue_n_xyc, once

def step_animate(screen, 
                 colors, 
                 t, 
                 center_x, 
                 center_y, 
                 clock, 
                 recrdr, 
                 draw_queue_n_xyc,
                 once,
                 positions_t_n_xy, 
                 n_hop_adjacency_t_h_n_n, 
                 ego_idx=None, 
                 model_pick_i=None,
                 ping_i=None,
                 pong_i=None,
                 scale=50,
                 drawAll=False):
    
    num_timesteps, nhops, num_nodes, _ = n_hop_adjacency_t_h_n_n.shape

    screen.fill((0, 0, 0))  # Clear the screen with black

    for n in range(num_nodes):
        x, y = positions_t_n_xy[t, n, :]
        skipDraw = False
        

        if ego_idx is None:
            indicies = n_hop_adjacency_t_h_n_n[t, 0, n].nonzero().squeeze()
            # pygame.draw.circle(screen, (255, 255, 255), (int(x*scale)+center_x, int(y*scale)+center_y), 5)
            color = colors[2] 

        else:
            if drawAll and (n != ego_idx):
                color = colors[2]
                indicies = n_hop_adjacency_t_h_n_n[t, 0, n].squeeze().nonzero()

            elif n == ego_idx:
                indicies = n_hop_adjacency_t_h_n_n[t, 0, n].squeeze().nonzero()
                # pygame.draw.circle(screen, (102, 255, 102), (int(x*scale)+center_x, int(y*scale)+center_y), 5)
                color = colors[1] 

            elif n in n_hop_adjacency_t_h_n_n[t, :nhops-1, ego_idx].squeeze().nonzero():
                indicies = n_hop_adjacency_t_h_n_n[t, 0, n].squeeze().nonzero()
                color = colors[2]

            else:
                color = colors[2]
                skipDraw = True

            if n == ping_i:
                color = colors[0]
            elif n == pong_i:
                color = colors[3]
            elif n == model_pick_i:
                color = colors[4]
        
        if (not skipDraw):
            if indicies.ndim == 0:
                continue
            for inx in indicies:
                if inx != n:
                    x2, y2 = positions_t_n_xy[t, inx.item(), :]
                    pygame.draw.line(screen, (255, 255, 255), 
                                        (int(x*scale)+center_x, int(y*scale)+center_y), 
                                        (int(x2*scale)+center_x, int(y2*scale)+center_y), 1)
            # pygame.draw.circle(screen, color, (int(x*scale)+center_x, int(y*scale)+center_y), 5)
        if ((model_pick_i is not None) and n == ego_idx):
            x2, y2 = positions_t_n_xy[t, model_pick_i, :].squeeze()
            pygame.draw.line(screen, colors[1], 
                                (int(x*scale)+center_x, int(y*scale)+center_y), 
                                (int(x2*scale)+center_x, int(y2*scale)+center_y), 1)

        draw_queue_n_xyc.append((int(x*scale)+center_x, int(y*scale)+center_y, color))
    
    for xyc in draw_queue_n_xyc:
        pygame.draw.circle(screen, xyc[2], (xyc[0], xyc[1]), 5)
    draw_queue_n_xyc.clear()


    if (t + 1) % num_timesteps == 0 and not once:
        once = True
        print("Saved")
        # recrdr.save()
        # recrdr.write(screen)
        pygame.quit()
            
    t = (t + 1) % num_timesteps
    recrdr.write(screen)
    # if once == False:
        # recrdr.click(screen)
    pygame.display.flip()  # Update the display
    clock.tick(20) 

    # event = pygame.event.wait()
    # waiting_for_key = True
    # while waiting_for_key:
    #     event = pygame.event.wait() # This line blocks until an event occurs

    #     if event.type == pygame.KEYDOWN:
    #         print(f"Key pressed: {pygame.key.name(event.key)}")
    #         waiting_for_key = False # Exit the loop after a key is pressed
    return t