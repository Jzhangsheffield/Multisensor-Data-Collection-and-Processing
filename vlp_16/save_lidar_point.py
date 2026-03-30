import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs_py.point_cloud2 as pc2


import os
from datetime import datetime
import time
import numpy as np



class SaveLidarPoint(Node):
    def __init__(self):
        super().__init__('save_lidar_point')

        # 配置参数：保存路径
        self.declare_parameter('save_dir', '/home/uos/multi_sensor_sync_save_data_test/velodyne')
        base_save_dir = self.get_parameter('save_dir').get_parameter_value().string_value

        # 创建 lidar_point 的订阅器
        self.cloud_sub = self.create_subscription(PointCloud2, '/velodyne_points', self.cloud_callback, 10)

        date = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.save_dir_cloud = os.path.join(base_save_dir, date)
        os.makedirs(self.save_dir_cloud, exist_ok=True)

        self.get_logger().info('SaveLidarPoint Node started')

    def cloud_callback(self, cloud):
        timestamp = str(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

        # 保存点云为 CSV
        try:
            points = pc2.read_points(cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True)
            point_array = np.array(list(points))
            cloud_path = os.path.join(self.save_dir_cloud, f"{timestamp}_cloud.csv")
            np.savetxt(cloud_path, point_array, delimiter=',', header='x,y,z,intensity', comments='')
        except Exception as e:
            self.get_logger().error(f"Failed to save point cloud: {e}")

        self.get_logger().info(f"Saved cloud data with timestamp: {timestamp}")

def main(args=None):
    rclpy.init(args=args)
    node = SaveLidarPoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()  