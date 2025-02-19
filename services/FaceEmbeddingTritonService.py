# services/face_embedding_service.py

import numpy as np
import cv2
from services.triton_model_wrapper import TritonModelWrapper

class FaceEmbeddingTritonService:
    """
    Uses the InceptionResnetV1 ONNX for face embedding.
    """
    def __init__(
        self,
        model_name="face_embedding",
        triton_url="localhost:8001",
        input_name="input",
        output_name="output_embedding",
        input_shape=(1,3,160,160)
    ):
        self.model_wrapper = TritonModelWrapper(
            model_name=model_name,
            triton_url=triton_url,
            input_metadata={
                "name": input_name,
                "datatype": "FP32",
                "shape": [1,3,160,160]
            },
            output_metadata=[
                {"name": output_name, "datatype": "FP32"}
            ],
            preprocess_fn=self._preprocess,
            postprocess_fn=self._postprocess
        )

    def compute_embedding(self, face_image):
        return self.model_wrapper.infer(face_image)

    def _preprocess(self, face_image):
        """
        BGR->RGB, resize to (160,160), normalize to [-1,1], reorder => (1,3,160,160).
        """
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (160,160))
        # (pixel - 127.5)/128 => [-1,1]
        arr = resized.astype(np.float32)
        arr = (arr - 127.5) / 128.0
        arr = np.transpose(arr, (2,0,1))  # (3,160,160)
        arr = np.expand_dims(arr, axis=0) # (1,3,160,160)
        return arr

    def _postprocess(self, outputs):
        """
        Expect a shape (1,512) => flatten to (512,).
        """
        embedding_batch = outputs[0]
        if embedding_batch is None or embedding_batch.shape[0] == 0:
            return None
        return embedding_batch[0]  # shape (512,)
