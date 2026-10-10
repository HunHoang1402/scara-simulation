from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'scara_sim'
MODEL_SRC = os.path.join('..', '..', '..', 'mujoco_sim')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # MuJoCo models live in <repo>/mujoco_sim; install them with the package
        ('share/' + package_name + '/models',
            glob(os.path.join(MODEL_SRC, '*.xml'))),
        ('share/' + package_name + '/models/meshes',
            glob(os.path.join(MODEL_SRC, 'meshes', '*.stl'))),
    ],
    install_requires=['setuptools', 'mujoco'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='ROS 2 and MuJoCo simulation package for Yamaha SCARA robot',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'mujoco_bridge = scara_sim.mujoco_bridge:main'
        ],
    },
)
