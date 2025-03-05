import cv2
import sys
import os
import re
import numpy as np
import tkinter as tk

def get_screen_resolution():
    """Get the current screen resolution using tkinter."""
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w, screen_h

def extract_part_number(filename):
    """Extract the part number from filename if present (e.g., 'part1')."""
    match = re.search(r'part(\d+)', filename, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def get_video_files(paths):
    """
    Given a list of paths (files or directories), return a sorted list
    of video files. If a path is a directory, it will search for common
    video file extensions. Files with 'part' in their name are sorted
    by their part number.
    """
    video_files = []
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    for path in paths:
        if os.path.isdir(path):
            # Look for files with video extensions in the directory.
            for file in os.listdir(path):
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(path, file))
        else:
            video_files.append(path)
    
    def sort_key(file):
        base = os.path.basename(file)
        part = extract_part_number(base)
        if part is not None:
            # Files with a part number come first and are sorted by that number.
            return (0, part, base)
        else:
            # Files without a part number are sorted alphabetically.
            return (1, base)
    
    video_files.sort(key=sort_key)
    return video_files

def main(video_files, speed_factor=5.0):
    current_index = 0
    fullscreen = False
    screen_width = None
    screen_height = None

    # This multiplier speeds up rewind.
    rewind_multiplier = 5.0

    # Create a resizable window.
    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)

    while True:
        if current_index < 0 or current_index >= len(video_files):
            print("No more videos to play.")
            break

        video_path = video_files[current_index]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            current_index += 1
            continue

        # Retrieve the original FPS; if unavailable, assume 30.
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30

        # We'll override the display FPS to 120 for consistency.
        display_fps = 120
        # (Note: We still use a very short delay so that the main loop runs fast.)
        delay = 1

        # Calculate the number of frames corresponding to a 5-second skip.
        skip_frames = int(5 * orig_fps)

        print(f"Playing video {current_index + 1}/{len(video_files)}: {video_path}")
        print(f"Original FPS: {orig_fps:.2f}, Display FPS: {display_fps:.2f}, Delay: {delay} ms (Speed factor: {speed_factor}x)")
        print(f"Rewind/Fast-forward jump step: {skip_frames} frames (~5 seconds)")
        # print("Controls:")
        # print("  Space: Pause/Unpause")
        # print("  f: Toggle fullscreen fit")
        # print("  j: Jump backward ~5 seconds")
        # print("  l: Jump forward ~5 seconds")
        # print("  F: Toggle continuous FAST FORWARD mode")
        # print("  R: Toggle continuous REWIND mode")
        # print("  d: Next video")
        # print("  a: Previous video")
        # print("  q: Quit")

        paused = False

        # Initialize continuous mode flags and accumulators.
        continuous_fast_forward = False
        continuous_rewind = False
        ff_accumulator = 0.0
        rw_accumulator = 0.0

        while cap.isOpened():
            if not paused:
                if continuous_rewind:
                    # For rewind, we update the accumulator and jump back more frames.
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    rw_accumulator += (speed_factor - 1)
                    # Multiply the skipped frames by the rewind multiplier.
                    skip_count = (1 + int(rw_accumulator)) * rewind_multiplier
                    rw_accumulator -= int(rw_accumulator)
                    new_frame = max(current_frame - skip_count, 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                    ret, frame = cap.read()
                    if not ret:
                        print("Reached beginning of video during rewind. Exiting rewind mode.")
                        continuous_rewind = False
                        continue

                else:
                    ret, frame = cap.read()
                    if not ret:
                        # End of current video: autoplay the next one.
                        print("Reached end of video segment. Autoplaying next video...")
                        if current_index < len(video_files) - 1:
                            current_index += 1
                            break
                        else:
                            print("This is the last video. Quitting...")
                            cap.release()
                            cv2.destroyAllWindows()
                            sys.exit(0)
                    
                    if continuous_fast_forward:
                        # In fast forward mode, accumulate extra frame skips.
                        ff_accumulator += (speed_factor - 1)
                        # Skip as many frames as accumulated.
                        while ff_accumulator >= 1:
                            ret_grab = cap.grab()
                            if not ret_grab:
                                break
                            ff_accumulator -= 1

                # Resize frame if in fullscreen mode.
                if fullscreen and screen_width is not None and screen_height is not None:
                    h, w = frame.shape[:2]
                    scale = min(screen_width / w, screen_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                    # Create a black background.
                    letterboxed = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                    # Center the resized frame.
                    x_offset = (screen_width - new_w) // 2
                    y_offset = (screen_height - new_h) // 2
                    letterboxed[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
                    cv2.imshow("Video Player", letterboxed)
                else:
                    cv2.imshow("Video Player", frame)

            # Wait for key press.
            key = cv2.waitKey(delay if not paused else 30) & 0xFF

            if key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Unpaused")
                continue

            elif key == ord('f'):
                # Toggle fullscreen "fit" mode.
                if not fullscreen:
                    screen_width, screen_height = get_screen_resolution()
                    cv2.setWindowProperty("Video Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    fullscreen = True
                    print("Fullscreen fit enabled")
                else:
                    cv2.setWindowProperty("Video Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    fullscreen = False
                    print("Fullscreen disabled")
                continue

            # Jump backward ~5 seconds.
            elif key == ord('j'):
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_frame = max(current_frame - skip_frames, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                print("Jumped backward ~5 seconds")
                # Reset continuous mode accumulators (if any).
                ff_accumulator = 0.0
                rw_accumulator = 0.0
                continue

            # Jump forward ~5 seconds.
            elif key == ord('l'):
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                new_frame = min(current_frame + skip_frames, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                print("Jumped forward ~5 seconds")
                ff_accumulator = 0.0
                rw_accumulator = 0.0
                continue

            # Toggle continuous FAST FORWARD mode.
            elif key == ord('F'):
                continuous_fast_forward = not continuous_fast_forward
                if continuous_fast_forward:
                    continuous_rewind = False  # Ensure only one mode is active.
                    ff_accumulator = 0.0
                    print("Continuous fast forward mode enabled")
                else:
                    print("Continuous fast forward mode disabled")
                continue

            # Toggle continuous REWIND mode.
            elif key == ord('R'):
                continuous_rewind = not continuous_rewind
                if continuous_rewind:
                    continuous_fast_forward = False
                    rw_accumulator = 0.0
                    print("Continuous rewind mode enabled")
                else:
                    print("Continuous rewind mode disabled")
                continue

            elif key == ord('d'):
                if current_index < len(video_files) - 1:
                    print("Switching to next video...")
                    current_index += 1
                else:
                    print("This is the last video.")
                break

            elif key == ord('a'):
                if current_index > 0:
                    print("Switching to previous video...")
                    current_index -= 1
                else:
                    print("This is the first video.")
                break

            elif key == ord('q'):
                print("Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

        cap.release()

    cv2.destroyAllWindows()
    print("No more videos to play.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py video_or_folder1 video_or_folder2 ...")
        sys.exit(1)

    video_files = get_video_files(sys.argv[1:])
    if not video_files:
        print("No video files found.")
        sys.exit(1)

    main(video_files)
