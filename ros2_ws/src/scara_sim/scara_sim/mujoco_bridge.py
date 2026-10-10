import os
# Native-Wayland GLFW can block in eglSwapBuffers and hang/segfault on exit; use the
# X11 build (XWayland) unless the user chose a variant (same fix as the panel).
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
import mujoco
import mujoco.viewer
import threading
import time


def default_model_path():
    return os.path.join(get_package_share_directory('scara_sim'),
                        'models', 'scara_real.xml')


class MujocoBridge(Node):
    def __init__(self):
        super().__init__('mujoco_bridge')

        # Defaults to the model installed with the package; override with
        # --ros-args -p model_path:=/abs/path/model.xml
        self.declare_parameter('model_path', default_model_path())
        model_path = self.get_parameter('model_path').value

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.get_logger().info(f"MuJoCo model loaded: {model_path}")

        # Resolve joint/actuator indices by name so the bridge works with any
        # model that uses these names
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        self.qpos_adr = []
        self.dof_adr = []
        self.act_idx = {}
        for name in self.joint_names:
            jnt = self.model.joint(name)
            self.qpos_adr.append(int(jnt.qposadr[0]))
            self.dof_adr.append(int(jnt.dofadr[0]))
        for a in range(self.model.nu):
            jid = self.model.actuator_trnid[a][0]
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname in self.joint_names:
                self.act_idx[jname] = a
        missing = set(self.joint_names) - set(self.act_idx)
        if missing:
            self.get_logger().warn(f"No actuator for joints: {sorted(missing)}")

        # Lock for thread safety between ROS callbacks and MuJoCo simulation loop
        self.lock = threading.Lock()

        # ROS 2 Publisher & Subscriber
        self.cmd_sub = self.create_subscription(
            JointState,
            '/joint_cmds',
            self.cmd_callback,
            10
        )
        self.state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.01, self.publish_state)  # 100 Hz

        # Start MuJoCo simulation thread; stop is set on shutdown
        self.stop = threading.Event()
        self.render_threads = set()
        self.sim_thread = threading.Thread(target=self.mujoco_loop, daemon=True)
        self.sim_thread.start()

    def cmd_callback(self, msg: JointState):
        with self.lock:
            for i, name in enumerate(msg.name):
                if name in self.act_idx and i < len(msg.position):
                    a = self.act_idx[name]
                    lo, hi = self.model.actuator_ctrlrange[a]
                    self.data.ctrl[a] = min(max(msg.position[i], lo), hi)

    def publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        with self.lock:
            msg.position = [float(self.data.qpos[a]) for a in self.qpos_adr]
            msg.velocity = [float(self.data.qvel[a]) for a in self.dof_adr]

        self.state_pub.publish(msg)

    def mujoco_loop(self):
        # Open viewer (it renders in its own thread; remember it to wait for it)
        before = set(threading.enumerate())
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.render_threads = set(threading.enumerate()) - before
            while viewer.is_running() and not self.stop.is_set() and rclpy.ok():
                step_start = time.time()

                with self.lock:
                    mujoco.mj_step(self.model, self.data)

                viewer.sync()

                # Sync timing
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        for t in self.render_threads:
            t.join(timeout=5)
        if not self.stop.is_set():
            # The user closed the viewer: end the node as well
            self.get_logger().info("Viewer closed, shutting down")
            rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MujocoBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        stuck = False
        if node is not None:
            # Close the viewer before the interpreter exits, otherwise glfw's atexit
            # terminate() runs under a live render loop (hang / segfault)
            node.stop.set()
            node.sim_thread.join(timeout=5)
            stuck = node.sim_thread.is_alive() or any(t.is_alive() for t in node.render_threads)
            node.destroy_node()
        rclpy.try_shutdown()
        if stuck:
            os._exit(0)


if __name__ == '__main__':
    main()
