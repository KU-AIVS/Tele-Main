# Tele-Main
Teleoperation-based VLA Data Collection Algorithm

<img src="/Assets/tele_main_setting.jpg"  width="700"/>


The teleporation-based VLA data construction algorithm focuses on efficiently generating high-quality robot learning data by combining ROS communication with data refinement techniques.

## Real-time teleperation integration over ROS
- Dual-arm control architecture: We adopt a control structure in which the motion of a robot arm without a gripper (master device) is transmitted via ROS to a gripper-equipped robot arm (controlled arm) and used to control it in real time.
- Intuitive data collection: By directly providing motions through teleoperation, the user can intuitively generate complex robot trajectories, which facilitates the collection of diverse, high-dimensional action data.

## Key Features
- Unified I/O: All data inputs and outputs-such as joint values, third-person RGB images, and keyboard inputs (for gripper control) are integrated and recorded at each step within the ROS environment.
- Multi-modality data storage: During task execution, the system simultaneously logs the robot arm's joint values (6-DoF joint angles) and third-person visual information (RGB images).
- Language instruction integration: For each episode, a corresponding natural language instruction is specified as the initial input, enabling the construction of vision-language-action mapping data that is essential for training VLA models.
- Automatic reset to initial state: At the end of each data collection episode, both rebot arms automatically return to their initial poses, keeping the starting state of subsequent episodes consistent and ensuring the reproducibility of data collection.

## Tutorial
1. Run tele_main.py to start teleoperation.

2. Directly move robot1 (LeaderArm) to control the behavior of robot2 (FollowerArm).

3. Gripper control and episode termination are performed via keyboard input (c: open gripper, z: close gripper, q: terminate).

4. Each robot action episode is saved as a single .npy file, which is internally organized as a step-wise dictionary.

Save format: episode_000_{instruction}.npy
000: episode index
{instruction}: language instruction assigned to that episode
