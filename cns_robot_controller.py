#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import Twist
import panda_py
from panda_py import controllers
import roboticstoolbox as rtb

class CNSControllerNode:
    def __init__(self):
        rospy.init_node("cns_robot_controller", anonymous=True)
        
        # 1. 机械臂初始化
        self.robot = rtb.models.Panda()
        self.k_null = 0.5  # 零空间优化任务的比例增益
        
        try:
            self.panda = panda_py.Panda("192.168.99.100")
            self.panda.move_to_start(speed_factor=0.05)
            self.q_pref = self.panda.q
            
            self.panda_ctrl = controllers.IntegratedVelocity()
            self.panda.start_controller(self.panda_ctrl)
            rospy.loginfo("Panda机械臂连接成功，已启动速度控制器。")
        except Exception as e:
            rospy.logerr(f"Panda硬件连接失败 ({e})，将切换至 模拟/测试 模式。")
            self.panda = None
            self.q_pref = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
            
        self.latest_twist = None
        self.last_msg_time = 0.0
        
        # 2. 手眼标定矩阵 (Camera Frame -> End-Effector Frame)
        # 根据参考脚本推断，Panda 末端到相机的旋转关系大概是: x_ee = -y_cam, y_ee = x_cam, z_ee = z_cam
        # 真实的平移偏差 t 请根据实际情况填入
        self.R_cam2ee = np.array([
            [ 0, -1,  0],
            [ 1,  0,  0],
            [ 0,  0,  1]
        ])
        self.t_cam2ee = np.zeros(3) # [tx, ty, tz]
        
        # 3. 订阅 CNS 输出的速度
        self.vel_sub = rospy.Subscriber("/cns/cmd_vel", Twist, self.vel_cb)
        rospy.loginfo("控制器已就绪，正在监听 /cns/cmd_vel ...")

    def vel_cb(self, msg):
        self.latest_twist = msg
        self.last_msg_time = rospy.get_time()

    def control_loop(self):
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            
            # 判断速度是否超时（放宽至 1.0 秒，适配 SuperGlue 等较慢的深度学习模型推理延迟）
            time_diff = current_time - self.last_msg_time
            if time_diff > 1.0 or self.latest_twist is None:
                rospy.logwarn_throttle(1.0, f"[超时断流] {time_diff:.2f} 秒未收到最新速度 (阈值 1.0s)，紧急刹车！")
                if self.panda is not None:
                    self.panda_ctrl.set_control(np.zeros(7))
                rate.sleep()
                continue
                
                
            # 1. 提取相机坐标系下的 6-DoF 速度
            v_cam = np.array([
                self.latest_twist.linear.x,
                self.latest_twist.linear.y,
                self.latest_twist.linear.z
            ])
            w_cam = np.array([
                self.latest_twist.angular.x,
                self.latest_twist.angular.y,
                self.latest_twist.angular.z
            ])
            
            # 2. 速度转换 (Camera Frame -> EE Frame)
            # 线速度 v_ee = R * v_cam + cross(t_ee_cam, R * w_cam)
            # 角速度 w_ee = R * w_cam
            v_ee_linear = self.R_cam2ee @ v_cam
            w_ee_angular = self.R_cam2ee @ w_cam
            v_ee_linear += np.cross(self.t_cam2ee, w_ee_angular)
            
            v_ee = np.concatenate([v_ee_linear, w_ee_angular])
            
            # 【关键修改 1】：全局比例缩放（阻尼）
            # 现在阻尼已经移交到了 predict_velocity.py 中处理，这里保持为 1.0
            velocity_gain = 1.0
            v_ee = v_ee * velocity_gain
            
            # 【关键修改 2】：严格安全限幅 (分别限制平移和旋转)
            max_ee_linear_vel = 0.05   # 限制线速度最大为 5 厘米/秒 (非常轻柔)
            max_ee_angular_vel = 0.2   # 限制角速度最大为 0.2 rad/s (约 11度/秒)
            
            if np.linalg.norm(v_ee[:3]) > max_ee_linear_vel:
                v_ee[:3] = (v_ee[:3] / np.linalg.norm(v_ee[:3])) * max_ee_linear_vel
                
            if np.linalg.norm(v_ee[3:]) > max_ee_angular_vel:
                v_ee[3:] = (v_ee[3:] / np.linalg.norm(v_ee[3:])) * max_ee_angular_vel
            # 3. 冗余度消解与关节速度计算
            if self.panda is not None:
                q_curr = self.panda.q
                
                # 计算末端在当前关节角下的解析雅可比矩阵
                J = self.robot.jacobe(q_curr)
                
                # 阻尼伪逆计算
                damping = 1e-4
                J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
                
                # 零空间投影矩阵
                P = np.eye(7) - J_pinv @ J
                
                # 零空间任务：拉拽关节向初始优选姿态靠拢，防止大关节漂移
                dq_null = -self.k_null * (q_curr - self.q_pref)
                
                # 逆运动学求解出目标关节速度
                dq = J_pinv @ v_ee + P @ dq_null
                
                # 限制为最大关节速度的 20%
                dq_max_limit = self.robot.qdlim[:7] * 0.2
                dq = np.clip(dq, -dq_max_limit, dq_max_limit)
                
                rospy.loginfo_throttle(0.5, f"[执行] 末端线速度 {np.linalg.norm(v_ee[:3]):.4f} m/s, 最大关节速度: {np.max(np.abs(dq)):.4f} rad/s")

                # 下发关节速度给机械臂
                self.panda_ctrl.set_control(dq)
                
            rate.sleep()

if __name__ == "__main__":
    try:
        controller = CNSControllerNode()
        controller.control_loop()
    except rospy.ROSInterruptException:
        pass
