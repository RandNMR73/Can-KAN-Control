from urdfpy import URDF
import numpy as np

# Load the URDF file
robot = URDF.load('arm.urdf')

# Cache the static transformations and joint data
def preprocess_kinematics(robot):
    # Map from child link name to joint
    link_to_joint = {joint.child: joint for joint in robot.joints}
    
    # Cache static transformations (joint.origin) for each joint
    joint_to_static_transform = {}
    for joint in robot.joints:
        if joint.origin is not None:
            static_transform = joint.origin
        else:
            static_transform = np.eye(4)  # Identity for joints with no origin
        joint_to_static_transform[joint.name] = static_transform

    return link_to_joint, joint_to_static_transform

# Helper function to compute a rotation matrix for a given axis and angle
def get_rotation_matrix(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    R = np.array([
        [c + x**2 * (1 - c), x*y*(1 - c) - z*s, x*z*(1 - c) + y*s],
        [y*x*(1 - c) + z*s, c + y**2 * (1 - c), y*z*(1 - c) - x*s],
        [z*x*(1 - c) - y*s, z*y*(1 - c) + x*s, c + z**2 * (1 - c)]
    ])
    # Convert 3x3 rotation matrix to 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    return T

# Optimized forward kinematics computation
def compute_forward_kinematics_fast(robot, joint_angles, link_to_joint, joint_to_static_transform):
    T = np.eye(4)  # Start with identity matrix (base frame)
    for joint in robot.joints:
        # Get the static transformation for this joint
        T_static = joint_to_static_transform[joint.name]

        # Get the dynamic transformation (joint rotation)
        if joint.joint_type in ['revolute', 'continuous']:
            angle = joint_angles.get(joint.name, 0)  # Default to 0 if no angle is given
            T_dynamic = get_rotation_matrix(joint.axis, angle)
        else:
            T_dynamic = np.eye(4)  # Fixed joint has no dynamic transformation

        # Combine static and dynamic transformations
        T = T @ T_static @ T_dynamic

    return T

# Preprocess the robot to cache static transformations
link_to_joint, joint_to_static_transform = preprocess_kinematics(robot)

def fk(t1, t2, t3, t4, t5):
    joint_angles = {"Revolute 2": 1, "Revolute 3": 1, "Revolute 5": 1, "Revolute 6": 1, "Revolute 7": 1}
    T_base_to_gripper = compute_forward_kinematics_fast(robot, joint_angles, link_to_joint, joint_to_static_transform)

print("Transformation Matrix (Base to Gripper):")
print(T_base_to_gripper)
