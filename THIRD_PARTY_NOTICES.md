# Third-Party Components

This project installs the following Python packages from PyPI:

- MediaPipe: https://github.com/google-ai-edge/mediapipe
- OpenCV: https://github.com/opencv/opencv
- NumPy: https://github.com/numpy/numpy

The setup script downloads the MediaPipe Hand Landmarker model from Google's
versioned model storage URL:

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

The model is not committed to this repository. Consult each upstream project
and model distribution for its applicable license and attribution terms.
