# Tele-Main
Teleoperation-based VLA Data Collection Algorithm

The teleporation-based VLA data construction algorithm focuses on efficiently generating high-quality robot learning data by combining ROS communication with data refinement techniques.

## Real-time teleperation integration over ROS
- Dual-arm control architecture: We adopt a control structure in which the motion of a robot arm without a gripper (master device) is transmitted via ROS to a gripper-equipped robot arm (controlled arm) and used to control it in real time.
- Intuitive data collection: By directly providing motions through teleoperation, the user can intuitively generate complex robot trajectories, which facilitates the collection of diverse, high-dimensional action data.
