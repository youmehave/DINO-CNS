import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import rospy
from geometry_msgs.msg import Twist

# 导入 CNS 核心库
from cns.benchmark.pipeline import CorrespondenceBasedPipeline

def main():
    target_img_path = "desired_image_d435i.png"
    pipeline_config = "pipeline.json"

    # 1. 检查目标照片和配置文件是否存在
    if not os.path.exists(target_img_path):
        print(f"[ERROR] 找不到目标照片 {target_img_path}，请先运行上一个脚本拍摄！")
        sys.exit(1)
    
    if not os.path.exists(pipeline_config):
        print(f"[ERROR] 找不到配置文件 {pipeline_config}！")
        sys.exit(1)

    # 初始化 ROS 节点和 Publisher
    rospy.init_node('cns_velocity_predictor', anonymous=True)
    vel_pub = rospy.Publisher('/cns/cmd_vel', Twist, queue_size=10)
    print("[INFO] ROS节点已启动，将速度发布至 /cns/cmd_vel 主题")

    print("[INFO] 正在初始化模型与流水线 (这可能需要几秒钟)...")
    # 2. 从 pipeline.json 优雅地加载流水线 (内部包含了特征提取器和图神经网络模型)
    # 注意：确保 pipeline.json 中的 "checkpoint" 路径 (如 checkpoints/cns.pth) 是正确的！
    try:
        model_pipeline = CorrespondenceBasedPipeline.from_file(pipeline_config)
    except Exception as e:
        print(f"[ERROR] 模型加载失败，请检查模型权重 cns.pth 是否存在！错误信息: {e}")
        sys.exit(1)

    # 3. 读取目标图片并设置
    # OpenCV 默认以 BGR 格式读取，符合流水线要求
    target_img = cv2.imread(target_img_path)
    # 假设目标距离相机粗略估计为 0.5 米 (distance_prior)
    distance_prior = 0.5 
    model_pipeline.set_target(target_img, distance_prior)
    print("[INFO] 目标图片设置完成！")

    # 4. 初始化 RealSense D435i 相机
    print("[INFO] 正在启动 RealSense 相机...")
    rs_pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        rs_pipeline.start(config)
    except RuntimeError as e:
        print(f"[ERROR] 相机启动失败: {e}")
        sys.exit(1)

    print("=========================================================")
    print("  系统已就绪！")
    print("  模型将持续对比当前画面与目标画面，并输出 6-DoF 速度指令。")
    print("  按下 'q' 键退出程序。")
    print("=========================================================")

    try:
        while True:
            # 5. 读取当前相机帧
            frames = rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            current_image = np.asanyarray(color_frame.get_data())

            # 6. 将当前画面传入模型，获取计算出的 6 自由度速度
            # 速度格式为: [vx, vy, vz, wx, wy, wz]
            vel, data, timing = model_pipeline.get_control_rate(current_image)

            # [新增] 在源头全局缩放模型输出的速度大小 (例如 0.1 表示削弱 90%)
            velocity_gain = 0.1
            vel = [v * velocity_gain for v in vel]

            # 打印结果，保留三位小数
            vel_str = ", ".join([f"{v: .3f}" for v in vel])
            print(f"[模型输出] 6-DoF 速度 (vx, vy, vz, wx, wy, wz): [{vel_str}]")

            # 7. 构造 ROS Twist 消息并发布
            twist_msg = Twist()
            twist_msg.linear.x = float(vel[0])
            twist_msg.linear.y = float(vel[1])
            twist_msg.linear.z = float(vel[2])
            twist_msg.angular.x = float(vel[3])
            twist_msg.angular.y = float(vel[4])
            twist_msg.angular.z = float(vel[5])
            vel_pub.publish(twist_msg)

            # 8. 显示当前视野（如果配置文件里设置了 MATCH 还能看到匹配的特征线）
            cv2.imshow("RealSense Live Feed - Press 'q' to quit", current_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 程序退出。")
                break
                
    finally:
        rs_pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
