import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.signal import find_peaks, welch

from modules.utils.logger import get_logger

logger = get_logger(__name__)


###############################################################################
# Minimal TS-CAN–like Model Definition (with global pooling)
###############################################################################

class TSCANModel(nn.Module):
    """
    A simplified TS-CAN–style network for rPPG estimation.
    We do global average pooling over (T,H,W), so we can handle
    variable time lengths without shape mismatch.
    """

    def __init__(self, frames=160):
        """
        :param frames: Number of frames in the output waveform (e.g., 160).
        """
        super(TSCANModel, self).__init__()
        self.frames = frames

        # Motion branch: shallow 3D CNN
        self.motion_conv1 = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(3,3,3), padding=(1,1,1))
        self.motion_conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3,3,3), padding=(1,1,1))
        self.motion_fc = nn.Linear(16, 128)

        # Appearance branch
        self.appear_conv1 = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(3,3,3), padding=(1,1,1))
        self.appear_conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3,3,3), padding=(1,1,1))
        self.appear_fc = nn.Linear(16, 128)

        # Attention mechanism
        self.attention_fc = nn.Linear(128, 128)

        # Final rPPG regressor => [B, frames]
        self.rppg_fc = nn.Linear(128, self.frames)

    def forward(self, motion_clip, appear_clip):
        """
        :param motion_clip: [B, 3, T, H, W]
        :param appear_clip: [B, 3, T, H, W]
        :return: rPPG signal [B, frames]
        """

        # --- Motion branch ---
        x_m = F.relu(self.motion_conv1(motion_clip))   # => [B,8, T, H, W]
        x_m = F.relu(self.motion_conv2(x_m))           # => [B,16,T, H, W]
        x_m = F.adaptive_avg_pool3d(x_m, (1,1,1))      # => [B,16,1,1,1]
        x_m = x_m.view(x_m.size(0), -1)                # => [B,16]
        x_m = self.motion_fc(x_m)                      # => [B,128]

        # --- Appearance branch ---
        x_a = F.relu(self.appear_conv1(appear_clip))   # => [B,8, T, H, W]
        x_a = F.relu(self.appear_conv2(x_a))           # => [B,16,T, H, W]
        x_a = F.adaptive_avg_pool3d(x_a, (1,1,1))      # => [B,16,1,1,1]
        x_a = x_a.view(x_a.size(0), -1)                # => [B,16]
        x_a = self.appear_fc(x_a)                      # => [B,128]

        # --- Attention ---
        attn = torch.sigmoid(self.attention_fc(x_a))   # => [B,128]
        x = x_m * attn                                  # => [B,128]

        # --- rPPG Wave ---
        rppg = self.rppg_fc(x)                          # => [B, self.frames]
        return rppg


###############################################################################
# Utility Functions
###############################################################################

def bandpass_filter(signal, fs, low=0.7, high=4.0):
    """
    Basic bandpass filter for typical HR range (~42-240 BPM).
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(3, [low/(fs/2), high/(fs/2)], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def compute_heart_rate(signal, fs):
    """
    Compute heart rate in BPM from a 1D rPPG signal using Welch PSD.
    """
    f, pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    valid_idx = np.where((f >= 0.7) & (f <= 4.0))[0]
    if len(valid_idx) < 1:
        return None

    freqs = f[valid_idx]
    power = pxx[valid_idx]
    peak_idx = np.argmax(power)
    peak_freq = freqs[peak_idx]
    hr_bpm = peak_freq * 60.0
    return hr_bpm


def estimate_spo2_from_rgb(frames):
    """
    Experimental approach to estimate SpO2 from RGB frames
    using a naive ratio-of-ratios approach.
    """
    frames_array = np.array(frames, dtype=np.float32)  # [N, H, W, 3]
    if frames_array.ndim != 4 or frames_array.shape[-1] != 3:
        logger.warning("Invalid frame array shape for SpO2 estimation.")
        return None

    mean_vals = frames_array.mean(axis=(0,1,2))  # => [3]
    red_mean = mean_vals[0] + 1e-6
    green_mean = mean_vals[1] + 1e-6

    ratio = red_mean / green_mean
    baseline_ratio = 0.9
    diff = ratio - baseline_ratio
    spo2_est = 100.0 - 5.0 * (diff * 10.0)
    spo2_est = np.clip(spo2_est, 80, 100)
    return spo2_est


###############################################################################
# RPPGTSCan Class (Sliding Window)
###############################################################################

class RPPGTSCan:
    """
    TS-CAN with a sliding window approach:
      - Keep a rolling buffer of frames up to 'frames_window'.
      - If buffer exceeds frames_window, pop from the front (oldest).
      - estimate_heart_rate() uses the last 'frames_window' frames.
      - We do NOT clear the buffer automatically, so we can get
        continuous or frequent estimates.
    """

    def __init__(self, device='cpu', frames_window=160, sampling_rate=30):
        """
        :param device: 'cpu' or 'cuda'
        :param frames_window: number of frames in a chunk for TS-CAN
                              (e.g. 160 frames ~ ~5s at 30 FPS)
        :param sampling_rate: approximate frames per second
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.frames_window = frames_window
        self.sampling_rate = sampling_rate

        # TS-CAN model (global pooling variant)
        self.model = TSCANModel(frames=frames_window).to(self.device)
        self.model.eval()
        logger.info(f"Initialized TSCANModel (sliding window) with frames_window={frames_window}, device={self.device}")

        # Optional: load pretrained weights
        # self.model.load_state_dict(torch.load("path/to/tscan_pretrained.pth", map_location=self.device))

        # Buffers
        self.buffer_frames = []  # store face frames (all resized)
        self.buffer_times = []

    def process_face_frame(self, face_roi):
        """
        Accumulate face ROI frames for rPPG estimation (sliding window).
        1) Convert to RGB if needed.
        2) Resize to a fixed dimension for consistency.
        3) Append to buffer. If buffer > frames_window, pop oldest frames.
        """
        if face_roi.size == 0:
            logger.warning("Empty face ROI. Skipping frame.")
            return

        # Convert BGR -> RGB if needed
        if face_roi.shape[2] == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        # Resize to ensure consistent shape, e.g. 160x160
        TARGET_SIZE = 160
        face_resized = cv2.resize(face_roi, (TARGET_SIZE, TARGET_SIZE))

        self.buffer_frames.append(face_resized)
        self.buffer_times.append(time.time())

        # Sliding window: if we exceed frames_window, pop the oldest
        while len(self.buffer_frames) > self.frames_window:
            self.buffer_frames.pop(0)
            self.buffer_times.pop(0)

        logger.debug(f"Buffer size after adding: {len(self.buffer_frames)}")

    def estimate_heart_rate(self):
        """
        Run the TS-CAN model on the current buffer to get rPPG wave,
        then compute HR in BPM.
        We do NOT clear the buffer, so we can do frequent calls.
        """
        if len(self.buffer_frames) < self.frames_window:
            logger.warning(f"Not enough frames to estimate HR. "
                           f"Have {len(self.buffer_frames)}, need {self.frames_window}")
            return None

        logger.debug("Preparing data for TS-CAN inference...")

        frames_array = np.array(self.buffer_frames, dtype=np.float32)
        # frames_array shape: [frames_window, H, W, 3]

        # Motion branch: naive frame difference
        diffs = []
        for i in range(1, len(frames_array)):
            diff = frames_array[i] - frames_array[i-1]
            diffs.append(diff)
        diffs = np.stack(diffs, axis=0)         # => [frames_window-1, H, W, 3]
        appear_frames = frames_array[1:]        # => [frames_window-1, H, W, 3]
        T_m = diffs.shape[0]

        logger.debug(f"Motion frames shape: {diffs.shape}, Appearance frames shape: {appear_frames.shape}")

        # Optional second downsample for the 3D conv
        H2, W2 = 36, 36
        motion_data = []
        appear_data = []
        for i in range(T_m):
            m = cv2.resize(diffs[i], (W2, H2))
            a = cv2.resize(appear_frames[i], (W2, H2))
            motion_data.append(m)
            appear_data.append(a)

        motion_data = np.array(motion_data)  # => [T_m, H2, W2, 3]
        appear_data = np.array(appear_data)

        # [B=1, 3, T_m, H2, W2]
        motion_data = motion_data.transpose(3, 0, 1, 2)  # => [3, T_m, H2, W2]
        appear_data = appear_data.transpose(3, 0, 1, 2)

        motion_data = torch.from_numpy(motion_data).unsqueeze(0).float().to(self.device)
        appear_data = torch.from_numpy(appear_data).unsqueeze(0).float().to(self.device)

        logger.debug(f"motion_data shape: {motion_data.shape}, appear_data shape: {appear_data.shape}")

        with torch.no_grad():
            rppg = self.model(motion_data, appear_data)  # => [1, frames_window]
        rppg = rppg.squeeze(0).cpu().numpy()  # => [frames_window]

        logger.debug(f"Raw TS-CAN output shape: {rppg.shape}")

        fs = self.sampling_rate if self.sampling_rate > 0 else 30.0
        filtered_rppg = bandpass_filter(rppg, fs)
        hr_bpm = compute_heart_rate(filtered_rppg, fs)

        if hr_bpm is None:
            logger.warning("Could not detect a valid HR peak in the rPPG signal.")
        else:
            logger.info(f"Estimated Heart Rate: {hr_bpm:.2f} BPM")
        return hr_bpm

    def estimate_spo2(self):
        """
        Estimate SpO2 from the current buffer using a naive ratio-of-ratios.
        We do NOT clear the buffer, so repeated calls can be made.
        """
        if len(self.buffer_frames) < 30:
            logger.warning(f"Not enough frames to estimate SpO2. Have {len(self.buffer_frames)}, need at least 30.")
            return None

        logger.debug("Estimating SpO2 from last frames in buffer...")
        spo2 = estimate_spo2_from_rgb(self.buffer_frames)
        if spo2 is not None:
            logger.info(f"Estimated SpO2: {spo2:.2f}%")
        else:
            logger.warning("Failed to estimate SpO2.")
        return spo2

    def clear(self):
        """
        Clear the buffers if you want to start fresh.
        Not used in a continuous sliding-window scenario,
        but provided for convenience.
        """
        logger.debug(f"Clearing buffers. Had {len(self.buffer_frames)} frames.")
        self.buffer_frames.clear()
        self.buffer_times.clear()
