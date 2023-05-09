import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from mtcnn import MTCNN

class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector_node')
       
        self.bridge = CvBridge()

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(Image,'/color/image',self.image_callback,10)

        # Initialize the MTCNN detector
        self.detector = MTCNN()

    def image_callback(self, msg):
        # Convert ROS image message to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg)

        # Detect faces in the image
        faces = self.detector.detect_faces(image)

        # Draw bounding boxes around the detected faces
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)

        # Display the output image
        cv2.imshow('FACE_DETECTOR', image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    node = FaceDetectorNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
