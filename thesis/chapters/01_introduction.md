# Introduction

## Motivation

Multirotor drones are increasingly asked to do more than follow waypoints. In inspection, logistics and search tasks an operator often knows *what* the drone should find — "the red box", "the blue barrel next to the door" — but not *where* it is. Classical onboard perception answers a narrower question: an object detector such as YOLO recognises the classes it was trained on, and anything outside that list is invisible to it. Vision-language models (VLMs) remove that limit. They take an image and a free-text instruction and return an answer about the image, including the location of the described object, without retraining for each new object type.

The price is size and speed. Useful VLMs have billions of parameters; even the small ones need several gigabytes of memory and take hundreds of milliseconds to seconds per image on a desktop GPU, and longer on the embedded computers a small drone can carry. Much of the published work on VLM- or LLM-driven drones therefore runs the model on a ground station or in the cloud and streams the video to it, or uses large onboard computers. A drone that should work without a network link, indoors and on a small airframe needs the model on board, and it needs to stay safe in the seconds during which the model is still thinking.

## Problem statement

Three things are missing for a small drone that selects its goal with an onboard VLM.

1. **It is not clear which open VLM to use and what it costs.** Model families, sizes and quantisation levels differ widely in grounding accuracy, in how often they invent objects that are not there, and in latency. Benchmarks are usually reported on everyday photographs at full precision on server GPUs, not on drone views, not quantised and not with the latency budget of an 8 GB edge computer.
2. **The perception module is more than the model.** A box in an image is not a place to fly to. The module has to turn the box into a 3-D goal, cope with wrong or missing answers, decide where to look when the target is not in view, and keep the drone away from obstacles independently of the slow model. How these parts fit together on board, and which factors limit them, is rarely described end to end.
3. **Deployment on a standard quadrotor is undocumented.** A practical system runs on off-the-shelf parts: a PX4 flight controller, a companion computer and a depth camera. How the software is divided between them, how data flows between the modules, and how the result is tested before it flies, is mostly left out of research papers.

## Research questions

The thesis answers three research questions.

- **RQ1 — Onboard perception.** How does the perception module of a drone work with onboard processing, and which factors (computer, model, sensors, latency, memory, safety) determine whether it works?
- **RQ2 — Model choice.** Which small, locally run VLM gives the best results for finding a described object in the scene, and how do accuracy and latency change with model size, quantisation and image size?
- **RQ3 — Deployment.** How can the perception and control software be deployed on a standard quadrotor platform (PX4 flight controller, Jetson companion computer, RGB-D camera), and how do the possible architectures — the VLM on board (pipeline A) or on a ground station fed by a video stream (pipeline B) — compare?

## Objectives

1. To understand the software and hardware requirements of an onboard VLM-based perception module for a small quadrotor.
2. To integrate the software modules so that they can be deployed on the hardware: the same code runs in simulation and on the drone.
3. To evaluate the deployment, first in simulation and then in real flights in the lab.

## Approach and contributions

The work combines a benchmark, an open-source software stack and a closed-loop evaluation.

- **VLM benchmark (RQ2).** Open VLMs from six families (Qwen2.5-VL, Qwen3-VL, Qwen3.5, Gemma 4, LFM3-VL and others), from 0.8 B to 32 B parameters, were compared on photographs (RefCOCO, gRefCOCO) and on drone images (RefDrone) with the exact prompt and parser the drone uses, including quantisation (4-bit, 8-bit, 16-bit), image size, two-step strategies such as tiling, and the rate at which a model answers "not found" correctly when the object is absent (Chapter 4).
- **Perception and safety stack (RQ1, RQ3).** A Python stack in which the VLM only selects the goal: the box and the depth image give a 3-D goal, two of three answers must agree, a turn-stop-look search (a fixed sweep or an active-sensing search) finds the target, and a depth-camera guard with a reactive go-around keeps the drone clear of obstacles at 20 Hz regardless of the VLM. It talks to PX4 through MAVSDK in OFFBOARD mode (Chapter 3).
- **Closed-loop simulation (RQ1, RQ3).** 60 runs in PX4 software-in-the-loop with Gazebo, comparing the two searches in worlds with and without obstacles, and a list of the failure modes found on the way and their fixes (Chapter 5).
- **Deployment plan (RQ3).** The hardware, the two pipeline architectures and a staged test plan from a VLM bench test on the Jetson to lab flights (Chapter 6).

All code, logs and results are in the thesis repository.

## Structure of the thesis

Chapter 2 reviews related work: VLM- and LLM-based drone navigation, onboard perception with detectors and VLMs, edge computers, depth-based obstacle avoidance, active perception and indoor state estimation. Chapter 3 describes the requirements and the system design. Chapter 4 reports the VLM evaluation (RQ2), Chapter 5 the simulation, and Chapter 6 the deployment on the quadrotor. Chapter 7 discusses the answers to the research questions and the limitations, and Chapter 8 concludes.
