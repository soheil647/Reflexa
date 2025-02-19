# services/triton_model_wrapper.py
import tritonclient.grpc as triton_grpc
import numpy as np

class TritonModelWrapper:
    """
    A generic wrapper around the Triton inference server.

    - model_name: Name of the deployed model on Triton (e.g. 'face_detect').
    - triton_url: The Triton server URL (e.g. 'localhost:8001').
    - input_metadata: Dict describing the expected input shape and datatype, e.g.:
         {
             "name": "input",
             "datatype": "FP32",
             "shape": [1, 3, 320, 320]  # or you can handle shape dynamically
         }
    - output_metadata: List of dicts describing each output, e.g.:
         [
             {"name": "output_boxes", "datatype": "FP32"},
             {"name": "output_scores", "datatype": "FP32"}
         ]
    - preprocess_fn: Optional callable that takes raw data (e.g., a frame)
      and returns a NumPy array suitable for Triton input.
    - postprocess_fn: Optional callable that takes the list of raw output arrays
      from Triton and returns final predictions (e.g., bounding boxes).
    """

    def __init__(
        self,
        model_name: str,
        triton_url: str = "localhost:8001",
        input_metadata: dict = None,
        output_metadata: list = None,
        preprocess_fn=None,
        postprocess_fn=None,
    ):
        self.model_name = model_name
        self.triton_url = triton_url
        self.input_metadata = input_metadata or {}
        self.output_metadata = output_metadata or []
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

        # Create the Triton client
        self.client = triton_grpc.InferenceServerClient(url=self.triton_url)

    def infer(self, data):
        """
        Perform inference on `data`.

        Steps:
          1) Pre-process (if a preprocess_fn is provided).
          2) Create Triton inputs.
          3) Request inference from Triton.
          4) Collect outputs.
          5) Post-process (if a postprocess_fn is provided).
        """
        # 1) Pre-process
        if self.preprocess_fn:
            data = self.preprocess_fn(data)

        # 2) Build Triton inputs
        inputs = []
        inp = triton_grpc.InferInput(
            self.input_metadata["name"],
            data.shape,
            self.input_metadata["datatype"],
        )
        inp.set_data_from_numpy(data)
        inputs.append(inp)

        # 3) Build Triton outputs
        outputs = []
        for out_meta in self.output_metadata:
            outputs.append(triton_grpc.InferRequestedOutput(out_meta["name"]))

        # 4) Inference
        results = self.client.infer(
            self.model_name,
            inputs,
            outputs=outputs
        )

        # 5) Post-process
        out_arrays = []
        for out_meta in self.output_metadata:
            raw = results.as_numpy(out_meta["name"])
            out_arrays.append(raw)

        if self.postprocess_fn:
            return self.postprocess_fn(out_arrays)
        else:
            return out_arrays
