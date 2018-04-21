DeepStream Preview
==================
This is the EA release of DeepStream

Sample function
---------------
Use deepStream to implement the video decode and inference.

# Build the sample
1. Need CUDA 8.0 to build the sample.
2. Edit the Makefile, set the "CUDA_PATH" as the installation path of your CUDA library.
3. Processd to the sample folder, and type "make" to build the sample

# Run the sample
1. In the sample, some scripts, network models and videos are given, you can run the script directly.
2. If you want to test your own neural network, change the network definition in the shell script (network model and weights). Remember to edit the nvDecInfer.cpp, and set input name and output name of your network.
3. Testing H.264 videos
	[1]. Some videos are given under the directory of "data".
	[2]. If you want to test your own videos, change the video definition in the shell script.
	[3]. Note that the input video should be raw video stream, no container.

