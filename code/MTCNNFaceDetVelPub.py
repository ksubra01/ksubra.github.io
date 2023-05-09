import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
from mtcnn import MTCNN

class FaceTrackingNode(Node):
    def __init__(self):
        super().__init__('face_tracking')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image,'/go/image',self.image_callback,10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.detector = MTCNN()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        results = self.detector.detect_faces(cv_image)
        if results:
            bounding_box = results[0]['box']
            x, y, w, h = bounding_box
            center_x = x + w/2
            center_y = y + h/2
            height, width, channels = cv_image.shape
            error_x = center_x - width/2
            error_y = center_y - height/2
            print(error_x)
            twist_msg = Twist()
            twist_msg.linear.x = 0.07  # set linear speed
            print('Moving straight towards face without mask')
            twist_msg.angular.z = -0.0008 * error_x  # set angular speed based on error
            print('Taking Steer turn towards face without mask')
            self.publisher.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FaceTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
