# Conclusion

This thesis set out to put a vision-language model on board a small, standard quadrotor so that it can be told in plain language what to fly to, and to keep it safe while the model thinks. The design answer is to give the VLM only one job — goal selection by a bounding box — and to surround it with classical parts: depth turns the box into a 3-D goal, two of three answers must agree, a turn-stop-look search with an active-sensing variant finds targets out of view, and a depth-camera guard with a reactive go-around runs at 20 Hz independently of the model.

The benchmark of open VLMs on photographs and drone images showed that Qwen3-VL-4B-Instruct at 4-bit quantisation (3.3 GB) is the best choice for an 8 GB edge computer: as accurate on drone views as models seven to eight times its size, no less accurate than its 16-bit version, and fast. Its weakness, inventing objects that are not there, is handled by the consensus rule.

In 60 closed-loop runs in PX4 and Gazebo, the complete stack reached the named target in 57 runs without a collision. The active search needed fewer VLM answers when the fixed sweep looked in the wrong direction first, and the same number otherwise. Building the simulation exposed the practical factors on which onboard perception depends — latency against motion, shared GPU memory, frozen and late sensor data, occlusion inside the box — and the stack now checks for each of them.

The hardware — X500 V2 with Pixhawk 6C, Jetson Orin Nano Super 8 GB and RealSense D435i — is on order, with indoor positioning by visual-inertial odometry and a staged plan from a VLM bench test to lab flights. Its results will show how much of the simulated performance survives real depth, real latency and real lab scenes.

All code, data, logs and results of this thesis are in its public repository.
