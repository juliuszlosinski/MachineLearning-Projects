import os
import wget
import numpy as np
import cv2
import argparse

"""
Class used to describing input videos.
"""
class VideoDescriber:
    def __init__(self) -> None:
        """
        Saving paths to the videos and model.
        """
        self.path_to_yolov3_conf:str = "./utils/yolov3.cfg"
        self.path_to_yolov3_weights:str = "./utils/yolov3.weights"
        self.path_to_coco_classes:str = "./utils/coco_classes.txt"
    
    def init(self, logging=False)->bool:
        """
        Initializing all components.
        """
        # Downloading YOLO model and coco classes:
        if not os.path.exists(self.path_to_yolov3_conf):
            wget.download("https://raw.githubusercontent.com/arunponnusamy/object-detection-opencv/master/yolov3.cfg", out="utils/yolov3.cfg")
        if not os.path.exists(self.path_to_yolov3_weights):
            wget.download("https://pjreddie.com/media/files/yolov3.weights", out="utils/yolov3.weights")
        if not os.path.exists(self.path_to_coco_classes):
            wget.download("https://raw.githubusercontent.com/arunponnusamy/object-detection-opencv/master/yolov3.txt", out="utils/coco_classes.txt")
        print(f"LOG::DEBUG::All files downloaded!\n")
        
        # Getting classes:
        with open(self.path_to_coco_classes, "r") as f:
            self.classes = [s.strip() for s in f.readlines()]
        self.index_classes_map = {index:value for index, value in enumerate(self.classes)}
        if logging:
            print(f"LOG::DEBUG::All coco classes set!\n")
        
        # Generating color for each class:
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Creating YOLO CNN model:
        self.yolo_model = cv2.dnn.readNet(self.path_to_yolov3_weights, self.path_to_yolov3_conf)
        self.layers = self.yolo_model.getLayerNames()
        self.output_layers = [self.layers[i - 1] for i in self.yolo_model.getUnconnectedOutLayers()]
        if logging:
            print(f"LOG::DEBUG::YOLO CNN Model created!\n")
    
    def convert_video(self, path_to_input_video, path_to_output_video, debug=False):
        """
        Converting video to objects detected video (.mp4 -> .mp4v)
        """
        log_file = open("."+str(path_to_input_video).split(".")[1]+".log", "w+")
        video = cv2.VideoCapture(path_to_input_video)
        fps = video.get(cv2.CAP_PROP_FPS)
        ret, frame = video.read()
        cv2.imwrite("test.jpg", frame)
        image = cv2.imread("test.jpg")
        height, width, _ = image.shape
        size = (width, height)
        os.remove("test.jpg")
        if debug:
            print(f"Size of the frame: {size}")
        video = cv2.VideoCapture(path_to_input_video)
        current_frame = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(path_to_output_video, fourcc, fps, size)
        while True:
            ret, frame = video.read()
            if ret:
                in_path = f"./data/frame{current_frame}.jpg"
                out_path = f"./data/out_frame{current_frame}.jpg"
                if debug:
                    print(f"LOG::DEBUG::Creating {in_path}")
                cv2.imwrite(in_path, frame)
                bounding_boxes_info = self.predict_and_draw_image(in_path, out_path, debug)
                result = f"\nCurrent frame: {current_frame}\n"
                for box in bounding_boxes_info:
                    result+=self.classes[int(box[0])]+", "+str(box[1])+"\n"
                    result+=self.convert_bounding_box_to_string(box)
                log_file.write(f"{result}\n{32*'-'}\n")
                image = cv2.imread(out_path)
                out_video.write(image)
                current_frame += 1
                os.remove(out_path)
                os.remove(in_path)
            else:
                break  
        out_video.release()
        log_file.close()
    
    def convert_bounding_box_to_string(self, bounding_box):
        """
        Converting bounding box to string.
        """
        result = "Bounding box: \n"
        result+=f"[ Ci = {'%.2f' % float(bounding_box[0])} ]\n"
        result+=f"[ Pc = {'%.2f' % float(bounding_box[1])} ]\n"
        result+=f"[ Bx = {'%.2f' % float(bounding_box[2])} ]\n"
        result+=f"[ By = {'%.2f' % float(bounding_box[3])} ]\n"
        result+=f"[ Bw = {'%.2f' % float(bounding_box[4])} ]\n"
        result+=f"[ By = {'%.2f' % float(bounding_box[5])} ]\n"
        return result
    
    def print_bounding_box(self, bounding_box):
        """
        Priniting the bounding box of the image.
        """
        print(self.convert_bounding_box_to_string(bounding_box))
    
    def convert_image(self, path_to_image):
        """
        Converting image to BLOB format.
        """
        readed_image = cv2.imread(path_to_image)
        converted_image_dimension = (readed_image.shape[1], readed_image.shape[0])
        converted_image = cv2.dnn.blobFromImage(
            image=readed_image,
            scalefactor=1/255,  # Scaling. 
            size=(416,416),     # Resizing.
            mean=(0,0,0), 
            swapRB=True, 
            crop=False
        )
        return converted_image, converted_image_dimension, readed_image

    def sort_objects(self, objects_detected, column_id):
        """
        Sorting objects.
        """
        number_of_objects = len(objects_detected)
        sorted_objects = {}
        for object, i in zip(objects_detected, range(number_of_objects)):
            sorted_objects[i]=object[column_id]
        sorted_objects = sorted(sorted_objects.items(), key=lambda x: x[1], reverse=True)
        final_sorted_objects = []
        for object in sorted_objects:
            final_sorted_objects.append(list(objects_detected[object[0]]))
        return final_sorted_objects
    
    def get_classes(self, classes, column_id, class_id):
        """
        Getting classes.
        """
        result = []
        for record in classes:
            if record[column_id] == class_id:
                result.append(record)
        return result

    def perform_nms(self, objects_detected, debug=True):
        """
        Performing NMS.
        """
        objects_detected = np.array(self.sort_objects(objects_detected, column_id=1))
        objects_detected_supp = []
        try:
            for class_id in set(objects_detected[:, 0]):
                objects_class = np.array(self.get_classes(objects_detected, 0, class_id))
                while objects_class.shape[0]>0:
                    currently_most_confident = objects_class[0]
                    objects_detected_supp.append(currently_most_confident)
                    objects_class = objects_class[1:]
                    indices_to_delete = []
                    for id in range(objects_class.shape[0]):
                        other = objects_class[id]
                        iou = self.compute_iou(currently_most_confident, other)
                        if debug:
                            print(f"IOU: {'%.10f'%iou}")
                        if iou > 0.5:
                            indices_to_delete.append(id)
                    for delete_id in indices_to_delete:
                        objects_class = np.array(self.get_classes(objects_class, 0, delete_id))
        except IndexError:
            return objects_detected_supp
        return objects_detected_supp

    def compute_iou(self, bounding_box_1, bounding_box_2):
        """
        Computing IoU factor.
        """
        b1_x, b1_y, b1_width, b1_height = bounding_box_1[2:6]
        b2_x, b2_y, b2_width, b2_height = bounding_box_2[2:6]

        b1_x_min = b1_x
        b1_y_min = b1_y
        b1_x_max = b1_x + b1_width
        b1_y_max = b1_y + b1_height
        
        b2_x_min = b2_x
        b2_y_min = b2_y
        b2_x_max = b2_x + b2_width
        b2_y_max = b2_y + b2_height
        
        x_overlap = max(0, min(b1_x_max, b2_x_max) - max(b1_x_min, b2_x_min))
        y_overlap = max(0, min(b1_y_max, b2_y_max) - max(b1_y_min, b2_y_min))

        intersection = x_overlap * y_overlap
        union = b1_width * b1_height + b2_width * b2_height - intersection

        iou = intersection / union if union > 0 else 0
        return iou

    def draw_bounding_boxes(self, path_to_input_image, path_to_output_image, objects_detected):
        """
        Drawing/ displaying bounding boxes.
        """
        _, _, image = self.convert_image(path_to_input_image)
        for object in objects_detected:
            b_x = int(object[2])
            b_y = int(object[3])
            b_width = int(object[4])
            b_height = int(object[5])
            cv2.line(
                img=image,
                pt1=(b_x, b_y),
                pt2=(b_x+b_width, b_y),
                color=self.colors[int(object[0])],
                thickness=3
            )
            cv2.line(
                img=image,
                pt1=(b_x, b_y),
                pt2=(b_x, b_y+b_height),
                color=self.colors[int(object[0])],
                thickness=3
            )
            cv2.line(
                img=image,
                pt1=(b_x, b_y+b_height),
                pt2=(b_x+b_width, b_y+b_height),
                color=self.colors[int(object[0])],
                thickness=3
            )
            cv2.line(
                img=image,
                pt1=(b_x+b_width, b_y),
                pt2=(b_x+b_width, b_y+b_height),
                color=self.colors[int(object[0])],
                thickness=3
            )
            cv2.putText(
                img=image,
                text=self.classes[int(object[0])]+", "+str(object[1]),
                org=(int(object[2]), int(object[3])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=self.colors[int(object[0])],
                thickness=3
            )
        cv2.imwrite(path_to_output_image, image)

    def predict(self, path_to_image, debug=False):
        """
        Predicting obejct by using YOLO model.
        """
        image, image_dimension, _ = self.convert_image(path_to_image)
        self.yolo_model.setInput(image)
        outputs = self.yolo_model.forward(self.output_layers)
        objects_detected = []
        for output in outputs:
            for detection in output:
                classes_scores = detection[5:]
                box_width = int(detection[2] * image_dimension[0])
                box_height = int(detection[3] * image_dimension[1])
                box_x = int(detection[0] * image_dimension[0] - 0.5 * box_width)
                box_y = int(detection[1] * image_dimension[1] - 0.5 * box_height)
                class_ID = np.argmax(classes_scores)
                confidence = max(classes_scores)

                if confidence > 0.5:
                    bounding_box = np.zeros((6))
                    bounding_box[0] = class_ID
                    bounding_box[1] = confidence
                    bounding_box[2] = box_x
                    bounding_box[3] = box_y
                    bounding_box[4] = box_width
                    bounding_box[5] = box_height
                    if debug:
                        print(f"------------------------")
                        print(f"Class ID: {class_ID}")
                        print(f"Class name: {self.classes[class_ID]}")
                        print(f"Confidence: {confidence}")
                        print(f"Box x: {box_x}")
                        print(f"Box y: {box_y}")
                        print(f"Box width: {box_width}")
                        print(f"Box height: {box_height}")
                        self.print_bounding_box(bounding_box)
                        print(f"-------------------------")
                        print(f"Detected: {self.classes[class_ID]}, {confidence} confidence, ({bounding_box[2]}, {bounding_box[3]}), {bounding_box[4]} x {bounding_box[5]}")
                    objects_detected.append(bounding_box)
        objects_detected = self.perform_nms(objects_detected, debug)
        return objects_detected
        
    def predict_and_draw_image(self, path_to_input_image, path_to_output_image, debug=False):
        """
        Simple prediction and drawing on output image.
        """
        bouding_boxes = self.predict(path_to_input_image, debug)
        self.draw_bounding_boxes(path_to_input_image, path_to_output_image, bouding_boxes)
        return bouding_boxes

# 1. Creating argument parser for application.
video_converter_parser = argparse.ArgumentParser()
video_converter_parser.add_argument("-i", "--input")
video_converter_parser.add_argument("-o", "--output")
args = video_converter_parser.parse_args()

# 2. Creating Video Describer that is using YOLO CNN model.
video_decriber = VideoDescriber()
video_decriber.init(logging=True)

# 3. Checking if this is a test running.
if args.input == "test" or args.input == "test":
    video_decriber.convert_video("./data/short_dogs_and_cats.mp4", "./data/short_converted_dogs_and_cats.mp4v", True)
    video_decriber.convert_video("./data/dogs_and_cats.mp4", "./data/converted_dogs_and_cats.mp4v", True)
    video_decriber.predict_and_draw_image("./data/cat.jpg", "./data/out_cat.jpg", True)
    video_decriber.predict_and_draw_image("./data/dog.jpg", "./data/out_dog.jpg", True)
# 3. Running custom scenario.
else:
    video_decriber.convert_video(args.input, args.output)