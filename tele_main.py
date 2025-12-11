"""
Dobot의 NOVA2를 이용한 teleoperation system

구성 환경:
    robot1: Leader Arm
    robot2: Follower Arm

사용 방법:
    python tele_main.py

저장 형식:
    episode_000_{instruction}.npy

데이터 구조 (Dictionary):
    각 스텝 (step)은 아래 키를 포함하는 dictionary 형태로 저장됩니다.

    Attributes:
        image (numpy.ndarray): RGB image data (height, width, 3)
        joint (numpy.ndarray): 7자유도 관절 각도
        language_instruction (str): 수행 중인 작업에 대한 언어 지시어
"""

from pynput.keyboard import Listener
from threading import Lock

import numpy as np
import pyrealsense2 as rs
import sys
import math
import time
import os
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from dobot_msgs_v4.srv import GetAngle, GetPose, StartDrag, StopDrag, ServoJ, SpeedFactor, MovJ, ServoP

from pymodbus.client import ModbusSerialClient


USB_PORT = '/dev/ttyUSB0'


class NOVA2(Node):
    def __init__(self, use_gripper=True):
        super().__init__('robot_controller')

        self.logger = self.get_logger()

        # Leader Arm Client
        self.MovJ_robot1 = self.create_client(MovJ, '/robot1/nova2_1/dobot_bringup_ros2/srv/MovJ')
        self.StartDrag_robot1 = self.create_client(StartDrag, '/robot1/nova2_1/dobot_bringup_ros2/srv/StartDrag')
        self.StopDrag_robot1 = self.create_client(StopDrag, '/robot1/nova2_1/dobot_bringup_ros2/srv/StopDrag')

        # Follower Arm Client
        self.MovJ_robot2 = self.create_client(MovJ, '/robot2/nova2_2/dobot_bringup_ros2/srv/MovJ')
        self.ServoJ_robot2 = self.create_client(ServoJ, '/robot2/nova2_2/dobot_bringup_ros2/srv/ServoJ')
        self.get_angle_robot2 = self.create_client(GetAngle, '/robot2/nova2_2/dobot_bringup_ros2/srv/GetAngle')

        self.ros_clients = [
            self.MovJ_robot1, self.StartDrag_robot1, self.StopDrag_robot1,
            self.MovJ_robot2, self.ServoJ_robot2,
        ]
        self._wait_for_services()

        # leader arm 제어
        self.sub = self.create_subscription(
            JointState,
            '/robot1/joint_states_robot',
            self.leader_joint_callback,
            10
        )

        # follower arm 제어
        self.sub_robot2 = self.create_subscription(
            JointState,
            '/robot2/joint_states_robot',
            self.follower_joint_callback,
            1
        )

        self.last_follower_qpos_rad = None
        self.qpos_lock = Lock()

        self.following = False

        self.recording = False
        self.recorded_path = []
        self.recording_thread = None
        self.recording_frequency = 10  # 녹화 주기 (예: 10Hz)
        self.instruction = "Instruction"

        self.save_path = os.path.join(os.getcwd(), 'dataset')
        os.makedirs(self.save_path, exist_ok=True)
        self.logger.info(f"데이터셋 저장 경로: {self.save_path}")

        self.pipeline = None
        self.align = None

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()

            # 해상도 및 프레임 설정
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

            align_to = rs.stream.color
            self.align = rs.align(align_to)

            self.pipeline.start(config)
            self.logger.info("Realsense 카메라 연결 성공.")

        except Exception as e:
            self.logger.error(f"!!! Realsense 카메라 연결 실패: {e} !!!")
            self.pipeline = None

        self.last_log_time = time.time()

        self.gripper_state = 1.0
        self.gripper_lock = Lock()

        self.go_home_requested = False

        self.gripper_client = None
        if use_gripper:
            try:
                self.gripper_client = ModbusSerialClient(port=USB_PORT, baudrate=115200, stopbits=1, bytesize=8, parity='N', timeout=0.1)
                if not self.gripper_client.connect():
                    raise ConnectionError("Gripper 연결 실패")
                self.gripper_client.write_register(0x0100, 1)
                self.logger.info("Gripper 연결 및 활성화")
                self.open_gripper()
            except Exception as e:
                self.logger.error(f"Gripper 초기화 실패: {e}")
                self.gripper_client = None

        self.logger.info("입력 키: 'c' (닫기), 'z' (열기), 'q' (종료)")
        self.key_listener = Listener(on_press=self.on_press)
        self.key_listener.start()

    def _wait_for_services(self):
        for client in self.ros_clients:
            if not client.wait_for_service(timeout_sec=2.0):
                self.logger.error(f"'{client.srv_name}' 서비스 찾을 수 없음")
                raise RuntimeError(f"필수 서비스 '{client.srv_name}' 연결 실패")
        self.logger.info("*** 연결 완료")

    def _call_sync(self, client, request):
        if not client.service_is_ready():
            self.logger.error(f"서비스 '{client.srv_name}'가 준비되지 않음")
            return None

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        if future.done():
            try:
                return future.result()
            except Exception as e:
                self.logger.error(f"'{client.srv_name}' 서비스 호출 중 예외 발생: {e}")
                return None
        else:
            self.logger.error(f"'{client.srv_name}' 서비스 호출 시간 초과")
            return None

    def on_press(self, key):
        try:
            if key.char == 'c':
                self.logger.info("*** Gripper Close (c)")
                self.close_gripper()
                with self.gripper_lock:
                    self.gripper_state = 0.0
            elif key.char == 'z':
                self.logger.info("*** Gripper Open (z)")
                self.open_gripper()
                with self.gripper_lock:
                    self.gripper_state = 1.0
            elif key.char == 'q':
                if not self.go_home_requested:
                    self.logger.info("*** 'q' 입력 감지! 녹화 중단 및 원점 복귀...")
                    self.go_home_requested = True
        except AttributeError:
            pass

    """ Drag 제어 """
    def start_drag(self):
        self.logger.info("*** [Leader Arm] Drag 모드 시작 중...")
        req = StartDrag.Request()
        res = self._call_sync(self.StartDrag_robot1, req)
        if res and res.res == 0:
            self.logger.info("*** [Leader Arm] Drag 모드 활성화 완료")
            self.following = True
        else:
            self.logger.warn("!!! [Leader Arm] Drag 모드 활성화 실패")

    def stop_drag(self):
        self.logger.info("*** [Leader Arm] Drag 모드 해제 중...")
        req = StopDrag.Request()
        res = self._call_sync(self.StopDrag_robot1, req)
        if res and res.res == 0:
            self.logger.info("*** [Leader Arm] Drag 모드 해제 완료")
            self.following = False
        else:
            self.logger.warn("!!! [Leader Arm] Drag 모드 해제 실패")

    def leader_joint_callback(self, msg: JointState):
        """ Leader Arm 제어 """
        if not self.following:
            return

        joints_deg = [math.degrees(pos) for pos in msg.position][:6]
        now = time.time()

        # ServoJ
        req = ServoJ.Request()
        req.a, req.b, req.c, req.d, req.e, req.f = joints_deg
        req.param_value = []
        self.ServoJ_robot2.call_async(req)

        # 로그 출력
        if now - self.last_log_time > 1.0:
            self.logger.info(f'Following...: {joints_deg}')
            self.last_log_time = now

    def follower_joint_callback(self, msg: JointState):
        joints_rad = msg.position[:6]

        with self.qpos_lock:
            self.last_follower_qpos_rad = joints_rad

    def _recording_loop(self):
        """Dataset 수집"""
        self.logger.info(">>> Recoding...")

        while self.recording and self.last_follower_qpos_rad is None:
            self.logger.info("Follower Arm Waiting...")
            time.sleep(0.5)

        while self.recording:
            loop_start_time = time.time()

            with self.qpos_lock:
                current_qpos_rad = self.last_follower_qpos_rad
            with self.gripper_lock:
                current_gripper_state = self.gripper_state

            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                # depth_frame = aligned_frames.get_depth_frame()

                if not color_frame:
                    self.logger.warn("!!! 프레임 수신 실패")
                    continue

                rgb_image = np.asanyarray(color_frame.get_data())
                ret = True

            except Exception as e:
                self.logger.warn(f"Realsense 프레임 읽기 오류: {e}")
                ret = False
                rgb_image = None

            if ret and current_qpos_rad is not None:
                joint_7dim = list(current_qpos_rad) + [current_gripper_state]
                data_point = {
                    'image': rgb_image.copy(),
                    'joint': np.array(joint_7dim),
                    'language_instruction': self.instruction,
                }
                self.recorded_path.append(data_point)

            elif not ret:
                self.logger.warn("녹화 중 프레임 캡처 실패")

            loop_duration = time.time() - loop_start_time
            sleep_time = (1.0 / self.recording_frequency) - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start_recording(self):
        """데이터셋 수집"""
        if self.recording:
            self.logger.info("녹화 진행 중")
            return

        self.logger.info(">>> 녹화 시작...")
        self.recording = True
        self.recorded_path = []
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()

    def stop_recording(self):
        """데이터셋 수집 중지"""
        if not self.recording:
            self.logger.info("녹화 중 아님")
            return []

        self.logger.info("*** 녹화 중지")
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
            self.recording_thread = None

        self.logger.info(f"녹화 완료. 총 {len(self.recorded_path)}개의 step 저장")
        return self.recorded_path

    def open_gripper(self):
        if self.gripper_client:
            self.gripper_client.write_register(0x0103, 1000)

    def close_gripper(self):
        if self.gripper_client:
            self.gripper_client.write_register(0x0103, 0)

    def move_joints_abs(self, joints_deg: list):
        req = MovJ.Request()
        req.mode = True
        req.a, req.b, req.c, req.d, req.e, req.f = joints_deg
        self._call_sync(self.MovJ_robot1, req)
        self._call_sync(self.MovJ_robot2, req)

    def get_joint(self, as_rad=False):
        res = self._call_sync(self.get_angle_robot2, GetAngle.Request())
        if res and res.res == 0:
            angles_deg = [float(val) for val in res.robot_return.strip('{}').split(',')]
            if as_rad:
                return np.deg2rad(angles_deg).tolist()
            return angles_deg
        return None

    def wait_for_movement_to_complete(self, target_angles, timeout=15):
        start_time = time.time()

        target_rad = np.deg2rad(target_angles)

        while time.time() - start_time < timeout:
            current_qpos_rad = self.get_joint(as_rad=True)

            if current_qpos_rad is None:
                time.sleep(0.01)
                continue

            is_complete = np.all(np.abs(np.array(current_qpos_rad) - target_rad) < 0.05)

            if is_complete:
                return True

            time.sleep(0.01)

        self.logger.warn(f"목표 지점 도달 시간 초과")
        return False

    def _reset(self):
        self.logger.info("원점 복귀 중...")
        self.open_gripper()
        home_joints = [90.0, 0.0, 90.0, 0.0, -90.0, 90.0]
        self.move_joints_abs(home_joints)
        self.wait_for_movement_to_complete(home_joints)
        self.logger.info("원점 복귀 완료")


def main(args=None):
    rclpy.init(args=args)
    nova_control = NOVA2()

    """ 로봇 동작 시 중간에 멈춰서 정렬이 안 맞을 때 True """
    is_align = False

    spin_thread = threading.Thread(target=rclpy.spin, args=(nova_control,), daemon=True)
    spin_thread.start()

    time.sleep(0.5)

    if is_align:
        nova_control._reset()
        time.sleep(3)
    dataset = []

    try:
        nova_control.following = True
        nova_control.start_drag()
        nova_control.start_recording()

        print("=" * 50)
        print("데이터 수집 중... 'c' (닫기), 'z' (열기), 'q' (종료)")
        print("=" * 50)

        while rclpy.ok() and not nova_control.go_home_requested:
            time.sleep(0.1)

        print("'q' 입력 확인. 녹화 중지")
        nova_control.following = False
        nova_control.stop_drag()
        dataset = nova_control.stop_recording()
        nova_control._reset()

    except KeyboardInterrupt:
        print("\nCrtl+C 입력. 녹화 강제 중지...")

    finally:
        if nova_control.recording:
            dataset = nova_control.stop_recording()

        if dataset:
            print(f"*** 총 {len(dataset)}개의 프레임 수집")
            try:
                save_path = nova_control.save_path
                os.makedirs(save_path, exist_ok=True)

                max_episode_id = -1
                for filename in os.listdir(save_path):
                    if filename.startswith('episode_') and filename.endswith('.npy'):
                        try:
                            episode_id_str = filename.split('_')[1]
                            episode_id = int(episode_id_str)
                            if episode_id > max_episode_id:
                                max_episode_id = episode_id
                        except ValueError:
                            continue

                new_episode_id = max_episode_id + 1
                save_filename = os.path.join(save_path, f'episode_{new_episode_id:03d}_{nova_control.instruction}.npy')

                np_data_to_save = np.array(dataset, dtype=object)
                np.save(save_filename, np_data_to_save)

                nova_control.logger.info(f"'{os.path.abspath(save_filename)}' 파일로 저장 완료")

            except Exception as e:
                nova_control.logger.error(f"!!! 데이터셋 저장 실패: {e}")

            if nova_control.pipeline:
                nova_control.logger.info("Realsense 파이프라인 중지")
                nova_control.pipeline.stop()

        if nova_control.key_listener.is_alive():
            nova_control.key_listener.stop()

        print("ROS 노드 종료 중...")
        nova_control.destroy_node()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except Exception as e:
        print(f"Main 함수 실행 중 오류 발생: {e}")
    finally:
        if rclpy.ok():
            print("rclpy 종료")
            rclpy.shutdown()
