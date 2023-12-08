# beyond-tbb-code
Code for Beyond the Black Box 2023-2024

# setup
Get the model: 

```
pip install -q mediapipe==0.10.0
!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```
# running

Run without streaming info to Omniverse
```
python body-pose-client.py --no-socket
```


# Reference
- Google Pose Landmarker example: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python