# export_models.py
import sys
import torch
import os

# TorchVision detection
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# For face embedding, we'll use facenet-pytorch's InceptionResnetV1
from facenet_pytorch import InceptionResnetV1

def export_fasterrcnn_detector(
    output_path="face_detect.onnx",
    opset_version=11,
    image_size=640
):
    """
    Exports a Faster R-CNN model (ResNet50 FPN) pretrained on COCO to ONNX.
    We'll treat it as a 'face detector' for demonstration, though it detects general objects.
    """
    print("Loading Faster R-CNN (ResNet50-FPN) from torchvision...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # TorchVision detection models expect a LIST of images, each [C,H,W].
    # We'll create a dummy image of shape [3, image_size, image_size].
    dummy_input = torch.randn(3, image_size, image_size)
    dummy_input_list = [dummy_input]  # The model expects a list

    print(f"Exporting Faster R-CNN to {output_path} with opset {opset_version}...")

    # We'll do a small trick: The model's forward returns a list of dict outputs,
    # which ONNX export doesn't handle well by default. We'll define a wrapper.
    class DetectionModelWrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod
        def forward(self, x):
            # returns list[dict], but we can flatten it or keep just the boxes/scores
            outputs = self.mod(x)  # list of length batch_size
            # For simplicity, let's just return the first image's boxes + scores + labels
            # So we have a consistent output shape.
            return (
                outputs[0]["boxes"],
                outputs[0]["scores"],
                outputs[0]["labels"]
            )

    wrapped_model = DetectionModelWrapper(model)

    # Now export the wrapped model
    torch.onnx.export(
        wrapped_model,
        (dummy_input_list,),
        output_path,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch_size"},  # we can treat the list as batch dimension
            "boxes": {0: "num_boxes"},
            "scores": {0: "num_boxes"},
            "labels": {0: "num_boxes"}
        }
    )
    print(f"Exported Faster R-CNN model to {output_path}.")

def export_inception_face_embedding(
    output_path="face_embedding.onnx",
    opset_version=11,
    device="cpu"
):
    """
    Exports the InceptionResnetV1 (facenet-pytorch) model to ONNX format.
    """
    print("Loading InceptionResnetV1 (pretrained='vggface2')...")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Create dummy input for shape [1,3,160,160]
    dummy_input = torch.randn(1, 3, 160, 160, device=device)

    print(f"Exporting InceptionResnetV1 to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output_embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output_embedding': {0: 'batch_size'}
        },
        opset_version=opset_version
    )
    print(f"Exported InceptionResnetV1 to {output_path}")

def main():
    # 1) Export "face detector" (Faster R-CNN)
    export_fasterrcnn_detector(
        output_path="../model_repository/face_detect/1/face_detect.onnx",
        opset_version=11,
        image_size=640
    )

    # 2) Export face embedding (InceptionResnetV1)
    export_inception_face_embedding(
        output_path="../model_repository/face_embedding/1/face_embedding.onnx",
        opset_version=11,
        device="cpu"
    )

if __name__ == "__main__":
    main()
