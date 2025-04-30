import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory


class BallDetectorNode(Node):
    def __init__(self):
        super().__init__('ball_detector_node')
        self.publisher_ = self.create_publisher(Float32, 'ball_distance', 10)
        self.timer = self.create_timer(0.1, self.detect_ball)
        
        video_path = os.path.join(
            get_package_share_directory("ball_det_package"),
            "sample3.mp4"
        )
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.get_logger().error('No Video Opened')
            rclpy.shutdown()

    def publish_distance(self, distance):
        msg = Float32()
        msg.data = distance
        self.publisher_.publish(msg)

    def ball(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([1, 80, 20])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        kernel = np.ones((15, 15), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def field(self, frame):
        low_green = np.array([30, 30, 45])
        up_green = np.array([85, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        green_mask = cv2.inRange(hsv, low_green, up_green)
        green_mask = cv2.erode(green_mask, kernel, iterations=1)
        green_mask = cv2.dilate(green_mask, kernel, iterations=6)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return frame, np.zeros_like(green_mask)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        mask = np.zeros_like(green_mask)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        return cv2.bitwise_and(frame, frame, mask=mask), mask

    def find_first_last_orange(self, line_data, line_start):
        first_orange = None
        last_orange = None
        for idx in range(line_data.shape[0]):
            if line_data[idx] == 255:
                first_orange = line_start + idx
                break
        for idx in range(line_data.shape[0] - 1, -1, -1):
            if line_data[idx] == 255:
                last_orange = line_start + idx
                break
        return first_orange, last_orange

    def find_top_bottom_orange(self, column_data, col_start):
        top_orange = None
        bot_orange = None
        for idy in range(column_data.shape[0]):
            if column_data[idy] == 255:
                top_orange = col_start + idy
                break
        for idy in range(column_data.shape[0] - 1, -1, -1):
            if column_data[idy] == 255:
                bot_orange = col_start + idy
                break
        return top_orange, bot_orange

    def detect_ball(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info('Video ended')
            rclpy.shutdown()
            return

        seg_field, mask_field = self.field(frame)
        mask_ball = self.ball(seg_field)

        blurred = cv2.GaussianBlur(mask_ball, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 50, param1=50, param2=30, minRadius=1, maxRadius=1000)

        if circles is not None:
            for i in circles[0, :]:
                circles = np.uint16(np.around(circles))
                x, y, r = circles[0, 0][:3]
                stroke = int(1.1 * r)

                # horizontal line
                line_y = y
                line_x_start = max(0, x - stroke)
                line_x_end = min(mask_ball.shape[1] - 1, x + stroke)
                orange_hline = mask_ball[line_y, line_x_start:line_x_end]
                first_orange, last_orange = self.find_first_last_orange(orange_hline, line_x_start)

                if first_orange is None or last_orange is None:
                    continue

                total_x_pixel = last_orange - first_orange
                r_new = int(total_x_pixel / 2)
                x_new = first_orange + r_new

                line_x = x_new
                line_y_start = int(y) - int(r)
                line_y_end = int(y) + int(r)
                orange_vline = mask_ball[line_y_start:line_y_end, line_x]
                top_orange, bot_orange = self.find_top_bottom_orange(orange_vline, line_y_start)

                if top_orange is None or bot_orange is None:
                    continue

                total_y_pixel = abs(top_orange - bot_orange)
                y_new = bot_orange - int(total_y_pixel / 2)

                if y_new != y:
                    line_y = y_new
                    orange_hline = mask_ball[line_y, line_x_start:line_x_end]
                    first_orange, last_orange = self.find_first_last_orange(orange_hline, line_x_start)
                    if first_orange is None or last_orange is None:
                        continue
                    total_x_pixel = last_orange - first_orange
                    r_new = int(total_x_pixel / 2)
                    x_new = first_orange + r_new

                R = int(r_new * 1.5)
                x1, y1 = max(x_new - R, 0), max(y_new - R, 0)
                x2, y2 = min(x_new + R, frame.shape[1]), min(y_new + R, frame.shape[0])

                surrounding_field = mask_field[y1:y2, x1:x2]
                field_ratio = np.sum(surrounding_field == 255) / surrounding_field.size

                surrounding_ball = mask_ball[y1:y2, x1:x2]
                ball_ratio = np.sum(surrounding_ball == 255) / surrounding_ball.size

                if field_ratio > 0.16 and ball_ratio < 0.47:
                    actual_diameter = 0.13  # meter
                    focal_length = 714.1
                    detected_diameter = total_x_pixel
                    if detected_diameter == 0:
                        distance = 0
                    else:
                        distance = (actual_diameter * focal_length) / detected_diameter

                    msg = Float32()
                    msg.data = float(distance)
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'Distance: {distance:.2f} m')
                else:
                    continue

                break

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorNode()

    if not node.cap.isOpened():
        node.get_logger().error("Video file could not be opened")
        rclpy.shutdown()
        return

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

