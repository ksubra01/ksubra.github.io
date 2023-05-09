import sys
import cv2
import time
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg           import String
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from cv_bridge              import CvBridge, CvBridgeError
from tensorflow.keras.models import load_model
from geometry_msgs.msg import Twist



#print('hI')
class MaskDetector(Node):

    def __init__(self):
        super().__init__('mask_detector')
        #self.detection_window = detection_window  
        self.br = CvBridge()
        print("Subscribed to the video feed>>>>")
        self.subscription = self.create_subscription(Image,'/color/image', self.listener_callback, 10)  
        self.model = load_model(r'/home/ksubra25/Downloads/model_for_inference.h5')
        
        
        #### Publisher ####
        self.publisher = self.create_publisher(Image,'/go/image', 10)
        #self.point = Point()
        self.vel = Twist()
        
    
        
    def listener_callback(self,data):
        
        current_frame = self.br.imgmsg_to_cv2(data)
        send_frame = current_frame
        current_frame=cv2.resize(current_frame,(300,300))
        current_frame=cv2.cvtColor(current_frame,cv2.COLOR_BGR2RGB)
        self.new_frame = current_frame
        cv2.imshow("image",current_frame)
        current_frame= current_frame/255.0
        current_frame= np.expand_dims(current_frame,axis=0)
        
        predictions=self.model.predict(current_frame)
        class_idx=np.argmax(predictions)
        if class_idx==0:
            label='with mask'
        elif class_idx==1:
            label='Without mask'  
            self.publisher.publish(self.br.cv2_to_imgmsg(send_frame))
        else:
            label='no face detected'
        
           
        cv2.putText(self.new_frame,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
        cv2.imshow('Detection',self.new_frame)     
        
        
        
        cv2.waitKey(1)
        #cv2.distroyAllWindows()
        
def main(args=None):

    rclpy.init(args=args)
    image_subscriber = MaskDetector()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
