import os
import time
import cv2
import torch
from loguru import logger
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from pypylon import pylon
from datetime import datetime
from collections import defaultdict

# Ensure that OpenCV has CUDA support
if not cv2.cuda.getCudaEnabledDeviceCount():
    raise RuntimeError("CUDA device not available. Ensure OpenCV with CUDA support is installed.")

# Directory to save detected images
output_dir = "Storage"
os.makedirs(output_dir, exist_ok=True)
check_file_path = os.path.join(output_dir, "check_object.txt")

# Dictionary to store max objects per second
max_objects_per_second = defaultdict(int)

# Instantiate the pylon DeviceInfo object and use it to get the cameras
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
if len(devices) < 1:
    raise ValueError("Not enough cameras found")

# Create a camera object
camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))

# Open the camera
camera.Open()

camera.Width.Value = 3840
camera.Height.Value = 2622
camera.OffsetX.Value = 0
camera.OffsetY.Value = 0

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
# Converting to OpenCV BGR format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule
            print(trt_file)
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=True,
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

def save_max_object_counts():
    """Save max object counts once per second"""
    with open(check_file_path, "a") as f:
        for time_key, max_count in max_objects_per_second.items():
            f.write(f"{time_key}: {max_count} object(s)\n")
    max_objects_per_second.clear()

def main():
    # Thiết lập các tham số
    image_path = "/home/orin-nano-2/Documents/Python_Project/YOLOX/NG_Image_16-17-27_jpg.rf.a4ad3f00abde8a66bf0af6682cbeef29.jpg"
    experiment_name = None
    model_name = None
    exp_file = (
        "/home/orin-nano-2/Documents/Python_Project/YOLOX/exps/example/custom/yolox_s.py"
    )
    ckpt_file = "/YOLOX/last_epoch_ckpt.pth"
    device = "gpu"
    conf = 0.35
    nms = 0.3
    tsize = None
    fp16 = False
    legacy = False
    fuse = False
    trt = True  # Sử dụng TensorRT

    # Lấy thiết lập từ file mô tả
    exp = get_exp(exp_file, model_name)
    if experiment_name is None:
        experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, experiment_name)
    os.makedirs(file_name, exist_ok=True)

    logger.info(f"Running inference on device: {device}")
    logger.info(f"Using TensorRT: {trt}")

    exp.test_conf = conf
    exp.nmsthre = nms
    if tsize is not None:
        exp.test_size = (tsize, tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if device == "gpu":
        model.cuda()
        if fp16:
            model.half()
    model.eval()

    if not trt:
        if ckpt_file is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if trt:
        assert not fuse, "TensorRT model does not support model fusing!"
        print(file_name)
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model not found! Run conversion script first."
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT for inference")

    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder, device, fp16, legacy
    )

    pTime = 0
    save_interval = 1  # Save only one image per second
    last_save_time = time.time()

    while True:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Resize the image
            image = converter.Convert(grabResult)
            img = image.GetArray()

            gpu_img = cv2.cuda_GpuMat()  # CUDA GpuMat for holding the image on GPU
            gpu_img.upload(img)  # Upload image to GPU

            # CUDA-accelerated resize
            gpu_resized_img = cv2.cuda.resize(gpu_img, (1900, 1500))  # Adjust size as necessary

            # Download resized image back to CPU for PyTorch inference
            resized_img = gpu_resized_img.download()

            # Checking time
            curr_time = time.time()
            fps = 1 / (curr_time - pTime)
            pTime = curr_time

            try:
                # Run inference
                outputs, img_info = predictor.inference(resized_img)
                # Check if any bounding boxes are detected
                annotated_frame = predictor.visual(outputs[0], img_info, predictor.confthre)

                if outputs[0] is not None and len(outputs[0]) > 0:
                    object_status = "Object: Found"
                    num_objects = len(outputs[0])

                    now = datetime.now()
                    # Format current time to the second
                    time_key = now.strftime("%Y-%m-%d %H:%M:%S")

                    # Update max count for this second if current count is higher
                    if num_objects > max_objects_per_second[time_key]:
                        max_objects_per_second[time_key] = num_objects

                    # Save image only once per second
                    if curr_time - last_save_time >= save_interval:
                        last_save_time = curr_time

                        # Create directory structure based on year, month, and day
                        year_dir = os.path.join(output_dir, now.strftime("%Y"))
                        month_dir = os.path.join(year_dir, now.strftime("%m"))
                        day_dir = os.path.join(month_dir, now.strftime("%d"))
                        os.makedirs(day_dir, exist_ok=True)

                        # Save the image with the filename as hour-minute-second
                        file_name = now.strftime("%H:%M:%S") + ".jpg"
                        save_path = os.path.join(day_dir, file_name)
                        cv2.imwrite(save_path, annotated_frame)
                else:
                    object_status = "Object: Not found"

                # Display FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {int(fps)}",
                    (0, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (255, 255, 0),
                    3,
                )

                # Display object status below FPS
                cv2.putText(
                    annotated_frame,
                    object_status,
                    (0, 100),  # Adjusted y-coordinate to position text below FPS
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (255, 255, 0),
                    3,
                )

                # Display FPS and object status (optional CUDA for text overlay)
                gpu_annotated_frame = cv2.cuda_GpuMat()
                gpu_annotated_frame.upload(annotated_frame)

                # Display the frame
                display_frame = cv2.cuda.resize(gpu_annotated_frame, (1500, 800))
                cv2.imshow("YOLOX Inference", display_frame.download())

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    save_max_object_counts()
                    break

            except Exception as e:
                print(e)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
