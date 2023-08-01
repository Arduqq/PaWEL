# PaWEL: Capturing your Heart with Webcams
## Setup
1. Make sure you have Python installed on your computer. You can download the latest version of Python from the official website (https://www.python.org/) and follow the installation instructions for your operating system.
2. Make sure to run `pip install -r requirements.txt`.
3. Ensure that a fitting HaarCascade file is in your cloned directory.
## Running the Application
1. Open the GUI using `python VideoLab.py`
2. Press *Initiate Video Feed* while having a webcam attached to your computer. This will open a new window with your current video feed. Try to be in a stable environment with good lighting conditions. You can adjust yourself so the bounding box produces suitable results.
3. As soon as the conditions are okay, you may press *Start Face Recording* to lock the current bounding box for tracking. Wait for the data to find enough data points until it produces suitable BPM values. You can see a real-time plot by pressing *Show Real-Time Plot*.
4. When you have gathered enough of data, you may press *Stop Recording & Save Sata*. Your captured BPM data should be called `data.csv`.
## Evaluation
Capturing heart data through video streams can be challenging to parameterize. It's advised to use dedicated technology to measure ground truth values. Consider pre-processing yur video files to help the system deduce reliable color values.