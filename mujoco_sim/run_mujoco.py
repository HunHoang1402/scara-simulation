import os
# Native-Wayland GLFW can block in eglSwapBuffers and hang/segfault on exit; use the
# X11 build (XWayland) unless the user chose a variant (same fix as the panel).
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import math
import threading
import time

import mujoco
import mujoco.viewer

def main():
    # Get the current directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "scara_real.xml")

    # Load the MuJoCo model and data
    print(f"Loading model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Open the viewer
    print("Starting simulation... Press ESC in the viewer to exit.")
    before = set(threading.enumerate())
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # launch_passive renders in its own thread; remember it to wait for it below
        render_threads = set(threading.enumerate()) - before

        # Simulation loop
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            elapsed_time = time.time() - start_time

            # Example: Simple sine wave control for the joints
            if len(data.ctrl) >= 4:
                # Joint 1: Shoulder (swing)
                data.ctrl[0] = math.sin(elapsed_time) * 1.5
                
                # Joint 2: Elbow
                data.ctrl[1] = math.cos(elapsed_time * 1.5) * 1.5
                
                # Joint 3: Z-axis slide (up and down)
                data.ctrl[2] = math.sin(elapsed_time * 2) * 0.05
                
                # Joint 4: Tool rotation
                data.ctrl[3] = math.sin(elapsed_time * 3) * 3.14

            # Step physics
            mujoco.mj_step(model, data)

            # Synchronize with viewer
            viewer.sync()

            # Time keeping (MuJoCo default timestep is usually 0.002s)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Wait for the render loop to stop before glfw is torn down at exit; if it
    # does not, skip interpreter teardown (glfw's atexit terminate would crash)
    for t in render_threads:
        t.join(timeout=5)
    if any(t.is_alive() for t in render_threads):
        os._exit(0)


if __name__ == "__main__":
    main()
