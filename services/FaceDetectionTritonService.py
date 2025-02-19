# services/face_detection_service.py

import numpy as np
import cv2
from services.triton_model_wrapper import TritonModelWrapper

class FaceDetectionTritonService:
    """
    A 'face detection' service using the exported Faster R-CNN model from TorchVision.
    We'll treat the highest-score 'person' or bounding box as the face bounding box
    for demonstration, but it's not truly specialized for faces.
    """
    def __init__(
        self,
        model_name="face_detect",
        triton_url="localhost:8001",
        input_name="images",
        output_names=("boxes", "scores", "labels"),
        input_shape=(1,3,640,640),
        score_threshold=0.5
    ):
        self.score_threshold = score_threshold
        self.model_wrapper = TritonModelWrapper(
            model_name=model_name,
            triton_url=triton_url,
            input_metadata={
                "name": input_name,
                "datatype": "FP32",
                # We'll treat the batch dimension as 1
                "shape": [1, 3, 640, 640]
            },
            output_metadata=[
                {"name": output_names[0], "datatype": "FP32"},
                {"name": output_names[1], "datatype": "FP32"},
                {"name": output_names[2], "datatype": "FP32"},
            ],
            preprocess_fn=self._preprocess,
            postprocess_fn=self._postprocess
        )

    def detect_faces(self, frame):
        """
        Returns a list of bounding boxes [(x1,y1,x2,y2), ...]
        We'll just pick bounding boxes with score > threshold,
        ignoring the 'labels' for simplicity.
        """
        return self.model_wrapper.infer(frame)

    def _preprocess(self, frame):
        """
        Convert BGR->RGB, resize to (640,640), scale to [0,1], reorder => (1,3,640,640).
        TorchVision detection expects images in range [0,1].
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640,640))
        # scale to [0,1]
        arr = resized.astype(np.float32) / 255.0
        # reorder => (3,640,640)
        arr = np.transpose(arr, (2,0,1))
        # add batch => (1,3,640,640)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _postprocess(self, outputs):
        """
        outputs = [boxes, scores, labels] each of shape (N, ...)
        We'll filter by score_threshold and convert to integer coords.
        Return list of (x1,y1,x2,y2).
        """
        boxes = outputs[0]  # shape (num_boxes, 4)
        scores = outputs[1] # shape (num_boxes,)
        labels = outputs[2] # shape (num_boxes,)

        if boxes is None or scores is None:
            return []

        # Convert to Python arrays
        boxes = boxes.astype(np.float32)
        scores = scores.astype(np.float32)
        labels = labels.astype(np.float32)

        valid_boxes = []
        for i in range(len(scores)):
            score = scores[i]
            if score < self.score_threshold:
                continue
            # For demonstration, we ignore label check or assume label=1 => 'person'
            # You might want to check if labels[i] == 1 or some face class
            x1, y1, x2, y2 = boxes[i]
            # Round to int
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            valid_boxes.append((x1,y1,x2,y2))

        return valid_boxes
