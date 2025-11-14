"""
This file contains an example of how to use the Camera class to capture and display
"""

from camera.Camera import Camera


def main():
    camera = Camera(colour_resolution=(1920, 1080), depth_resolution=(1280, 720), fps=30, enable_colour=True, enable_depth=True, show_colour=True, show_depth=True)
    camera.start()

    try:
        while True:
            colour_frame, depth_frame = camera.get_frames()
            if colour_frame is not None and depth_frame is not None:
                # Process frames as needed
                pass
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()

if __name__ == "__main__":
    main()