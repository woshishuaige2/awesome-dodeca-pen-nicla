# Offline Video Mode: Modification Guide

This document explains the modifications made to enable offline video analysis in the Dodeca Pen project. These changes allow you to replace the live camera feed with a pre-recorded video file for reproducible research and easier collaboration.

## Summary of Changes

The modifications are **minimal and non-destructive**. No code has been deleted; instead, the system now supports both live camera and offline video modes through a simple parameter change.

## Modified Files

### 1. `Code/Computer_vision/run.py`

**Changes:**
- Added a new parameter `video_file` to the `start()` and `main()` functions
- When `video_file` is specified, the system uses `cv2.VideoCapture(video_file)` instead of `cv2.VideoCapture(cam_index)`
- All original live camera logic is preserved and used when `video_file=None`

**Key Code Section (Lines 132-158):**
```python
def start(headless: bool = True, cam_index: int = 0, video_file: str = None) -> None:
    """
    Main vision loop.
    
    Args:
        headless: If True, runs without displaying the video window
        cam_index: Camera index for live capture (ignored if video_file is specified)
        video_file: Path to offline video file. If specified, uses this instead of live camera.
    """
    global object_pose

    # >>> MODIFICATION: Support offline video input <<<
    if video_file is not None:
        print(f"[CV] Using offline video file: {video_file}")
        cap = cv2.VideoCapture(video_file)
    else:
        # >>> ORIGINAL: Live camera capture <<<
        cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        if video_file is not None:
            print(f"[CV] Could not open video file: {video_file}")
        else:
            print(f"[CV] Could not open camera index {cam_index}")
        return
```

### 2. `Code/IMU/app/app.py`

**Changes:**
- Added a configuration variable `OFFLINE_VIDEO_FILE` in the `main()` function (line 471)
- This variable is passed to the CV thread when it starts
- Setting `OFFLINE_VIDEO_FILE = None` uses live camera (default)
- Setting `OFFLINE_VIDEO_FILE = "path/to/video.mp4"` uses offline video

**Key Code Section (Lines 466-478):**
```python
# --- NEW: launch the CV loop in-process so the bridge can see object_pose ---
# >>> MODIFICATION: Support offline video input <<<
# To use offline video, set video_file parameter to the path of your video file
# Example: video_file="offline_test.mp4" or video_file="/path/to/your/video.mp4"
# Set to None to use live camera (default behavior)
OFFLINE_VIDEO_FILE = None  # <<< CHANGE THIS to enable offline mode

cv_thread = threading.Thread(
    target=cv_run.main, 
    kwargs={"headless": False, "cam_index": 0, "video_file": OFFLINE_VIDEO_FILE}, 
    daemon=True
)
cv_thread.start()
```

## How to Enable Offline Mode

### Step 1: Prepare Your Video File

Record a video of the Dodeca Pen with ArUco markers clearly visible. The video should be in a standard format (MP4, AVI, MOV, etc.) that OpenCV can read.

**Recommended Video Specifications:**
- **Format:** MP4 (H.264 codec)
- **Resolution:** 640x480 or higher
- **Frame Rate:** 30 fps or higher
- **Duration:** As needed for your analysis
- **Content:** Dodeca Pen with ArUco markers clearly visible throughout the video

### Step 2: Place the Video File

You have two options:

**Option A: Relative Path (Recommended)**
Place your video file in the `Code/Computer_vision/` directory and use a relative filename:
```
awesome-dodeca-pen/
├── Code/
│   ├── Computer_vision/
│   │   ├── offline_test.mp4  ← Place your video here
│   │   └── run.py
│   └── IMU/
│       └── app/
│           └── app.py
```

**Option B: Absolute Path**
Place your video file anywhere and use the full path.

### Step 3: Modify `app.py`

Open `Code/IMU/app/app.py` and locate line 471. Change:

```python
OFFLINE_VIDEO_FILE = None  # Live camera mode
```

To one of the following:

**For relative path:**
```python
OFFLINE_VIDEO_FILE = "offline_test.mp4"
```

**For absolute path:**
```python
OFFLINE_VIDEO_FILE = "/home/user/videos/my_recording.mp4"
```

**For relative path from Computer_vision directory:**
```python
OFFLINE_VIDEO_FILE = str(Path(__file__).resolve().parents[2] / "Computer_vision" / "offline_test.mp4")
```

### Step 4: Run the Application

Run the application as usual:
```bash
cd Code/IMU
python main.py
```

The system will now process the video file instead of the live camera feed. You should see a message in the console:
```
[CV] Using offline video file: offline_test.mp4
```

## Switching Back to Live Camera Mode

To return to live camera mode, simply change the line back to:
```python
OFFLINE_VIDEO_FILE = None
```

## Expected Placeholder Video Filename

If you follow the recommended approach, the expected filename is:

**`offline_test.mp4`**

Place this file in the `Code/Computer_vision/` directory.

## Video Format Requirements

The video file should be compatible with OpenCV's `cv2.VideoCapture()`. Most common video formats are supported, including:

- **MP4** (H.264, MPEG-4)
- **AVI** (various codecs)
- **MOV** (QuickTime)
- **MKV** (Matroska)

**Important Notes:**
- Ensure the video codec is installed on your system
- For best compatibility, use MP4 with H.264 codec
- The video should show the Dodeca Pen with ArUco markers clearly visible
- Lighting conditions should be similar to your live camera setup

## Verification

To verify the modification is working correctly:

1. Check the console output for the message: `[CV] Using offline video file: ...`
2. The CV window should display frames from your video file
3. The system should track ArUco markers and update the 3D visualization as usual
4. The video will loop continuously (OpenCV behavior) or you can modify the code to stop after one pass

## Troubleshooting

**Problem:** "Could not open video file"
- **Solution:** Check that the file path is correct and the file exists
- **Solution:** Verify the video format is supported by OpenCV
- **Solution:** Try using an absolute path instead of a relative path

**Problem:** Video plays but no markers are detected
- **Solution:** Ensure the ArUco markers are clearly visible in the video
- **Solution:** Check that the lighting and focus are adequate
- **Solution:** Verify the camera calibration parameters match the camera used to record the video

**Problem:** Video plays too fast or too slow
- **Solution:** The system processes frames as fast as possible. If timing is critical, you may need to add frame rate control in the `start()` function

## Next Steps

Once you have recorded a compatible video file:
1. Name it `offline_test.mp4` (or update the path in `app.py`)
2. Place it in the `Code/Computer_vision/` directory
3. Update line 471 in `Code/IMU/app/app.py` as described above
4. Run the application and verify the offline mode is working

The system will then process your video file frame by frame, allowing for reproducible analysis without requiring a live camera setup.
