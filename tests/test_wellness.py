# tests/test_wellness.py
import cv2
from modules.wellness.health_metrics import HealthMetricsAnalyzer

def test_health_metrics():
    image = cv2.imread("data/sample_face.jpg")
    analyzer = HealthMetricsAnalyzer()
    metrics = analyzer.analyze(image)
    assert "heart_rate" in metrics, "Heart rate not found in metrics"
    assert "skin_health" in metrics, "Skin health not found in metrics"
