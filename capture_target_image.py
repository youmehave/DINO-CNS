import pyrealsense2 as rs
import numpy as np
import cv2
import os

def main():
    print("[INFO] Initializing RealSense D435i camera...")
    
    # Configure color stream
    pipeline = rs.pipeline()
    config = rs.config()

    # We only need the RGB camera for the target image
    # Resolution 640x480 is commonly used in this project
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"[ERROR] Failed to start camera: {e}")
        print("Please check if the RealSense camera is connected properly.")
        return

    print("[INFO] Camera started successfully.")
    print("=====================================================")
    print("  操作说明：")
    print("  1. 移动机械臂，将相机对准你希望抓取的最终目标位置。")
    print("  2. 在弹出的视频窗口中确认画面。")
    print("  3. 按下 's' 键拍摄并保存目标图片 (desired_image.png)。")
    print("  4. 按下 'q' 键退出程序。")
    print("=====================================================")

    try:
        while True:
            # Wait for a coherent frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            # Convert images to numpy arrays (H, W, 3) BGR format
            color_image = np.asanyarray(color_frame.get_data())

            # Show images
            cv2.imshow('RealSense D435i - Press "s" to save, "q" to quit', color_image)
            
            key = cv2.waitKey(1)
            
            # Press 's' to save the image
            if key & 0xFF == ord('s'):
                save_path = "desired_image_d435i.png"
                cv2.imwrite(save_path, color_image)
                print(f"[SUCCESS] 目标图片已成功保存至: {os.path.abspath(save_path)}")
                break
                
            # Press 'q' or 'esc' to quit
            elif key & 0xFF == ord('q') or key == 27:
                print("[INFO] 用户取消拍摄，程序退出。")
                break

    finally:
        # Stop streaming and close windows
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
