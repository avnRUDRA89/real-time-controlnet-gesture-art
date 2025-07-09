# real-time-controlnet-gesture-art
Generate real-time AI art using ControlNet and Stable Diffusion, conditioned by webcam-based gesture and gender detection.
# Real-Time ControlNet Gesture Art Generator

This project uses **MediaPipe**, **ControlNet**, and **Stable Diffusion** to create **real-time generative AI art** that responds to **human body gestures** and **gender recognition** from webcam input. It fuses Canny edge detection with gesture-conditioned prompts to generate stunning visuals, overlaid live on the webcam feed.

---

## 🎯 Features

- 🎥 Real-time webcam input
- 🧍 Pose tracking using MediaPipe
- ✋ Gesture classification (arms raised, squat, hands on hips, etc.)
- 🧠 Gender detection from face crop
- 🎨 AI image generation via ControlNet + Stable Diffusion
- 🖼️ Dynamic prompt conditioning based on detected gesture + gender
- 🧩 Overlay of AI-generated visuals on the live video stream

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/avnRUDRA89/real-time-controlnet-gesture-art.git
cd real-time-controlnet-gesture-art
