import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import mediapipe as mp
from controlnet_aux import CannyDetector


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device.upper()}")


base_prompt = (
    "epic cosmic portrait of a majestic person standing like a hero,"
    "in the style of Ghibli animation,"
    "Disney Classic illustration, Tim Burton gothic aesthetic, "
    "Spider-Verse comic art,"
    "with Minecraft voxel pixelation" 
    "Tetris abstraction,"
    "Arcane (Fortiche) painterly realism,"
    "Borderlands cel-shading," 
    "influenced by Art Deco geometric luxury,"
    "Pop Art repetition and color bursts like Andy Warhol,"
    "layered with Cyberpunk neon urban glow,"
    
)

negative_prompt = (
    "nude, naked, nsfw, erotic, shirtless, cleavage, revealing, exposed, lingerie, underwear, lowres, "
    "bad anatomy, blurry, distorted face, extra limbs, cropped, watermark, poorly drawn, malformed"
)


print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype
).to(device)

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xFormers enabled.")
except Exception as e:
    print("Skipping xFormers:", e)

canny = CannyDetector()


mp_pose = mp.solutions.pose
pose_tracker = mp_pose.Pose(static_image_mode=False, model_complexity=1)

gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
GENDER_LIST = ["Male", "Female"]

def detect_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426337, 87.768914, 114.895847), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    return GENDER_LIST[gender_preds[0].argmax()].lower()

def detect_gesture(landmarks):
    y = lambda name: landmarks[mp_pose.PoseLandmark[name]].y
    x = lambda name: landmarks[mp_pose.PoseLandmark[name]].x
    get = lambda name: landmarks[mp_pose.PoseLandmark[name]]

    nose_y = y("NOSE")
    lw_y, rw_y = y("LEFT_WRIST"), y("RIGHT_WRIST")
    lw_x, rw_x = x("LEFT_WRIST"), x("RIGHT_WRIST")
    lh_x, rh_x = x("LEFT_SHOULDER"), x("RIGHT_SHOULDER")
    l_knee_y = y("LEFT_KNEE")
    r_knee_y = y("RIGHT_KNEE")
    l_hip_y = y("LEFT_HIP")
    r_hip_y = y("RIGHT_HIP")

    if lw_y < nose_y and rw_y < nose_y:
        return "arms_raised"
    if lw_y < nose_y and rw_y > nose_y:
        return "left_up"
    if rw_y < nose_y and lw_y > nose_y:
        return "right_up"
    if abs(lw_y - rw_y) < 0.05 and abs(lw_x - rw_x) > 0.25:
        return "arms_out"
    if l_knee_y < l_hip_y - 0.1 and r_knee_y < r_hip_y - 0.1:
        return "squat"
    if lw_x < lh_x - 0.2 and rw_x < rh_x - 0.2:
        return "hands_on_hips"
    if rw_x > rh_x + 0.3:
        return "right_forward"
    if lw_x < lh_x - 0.3:
        return "left_forward"
    return "neutral"

gesture_modifiers = {
    "neutral": "",
    "arms_raised": ", radiant energy columns ascend as divine power surges upward",
    "left_up": ", mystic energy spirals leftward from raised hand",
    "right_up": ", glowing comet trails flow from raised right arm",
    "arms_out": ", majestic wings of light expand outward from shoulders",
    "squat": ", grounded in warrior stance, aura intensifies around the feet",
    "hands_on_hips": ", regal pose of command, celestial crown glows",
    "right_forward": ", cosmic beam channels forward from outstretched right arm",
    "left_forward": ", arcane portal opens from extended left hand"
}

gender_modifiers = {
    "male": ", masculine celestial warrior, glowing beard, divine armor",
    "female": ", ethereal cosmic goddess, flowing hair, celestial gown",
    "neutral": ", androgynous cosmic spirit, fluid radiant aura"
}


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")
print("📡 Streaming AI visuals... Press Q to quit.")

prev_time = 0
fps_limit = 30.0  

output_width = 1920  
output_height = 1080  

while True:
    current_time = time.time()
    elapsed = current_time - prev_time

    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed")
        break

    if elapsed >= 1.0 / fps_limit:
        prev_time = current_time

        frame_resized = cv2.resize(frame, (512, 512))  
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        
        face_crop = frame_resized[100:300, 150:350]  
        try:
            gender = detect_gender(face_crop)
        except:
            gender = "neutral"

        
        results = pose_tracker.process(rgb_frame)
        gesture = "neutral"
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            gesture = detect_gesture(lm)

        
        dynamic_prompt = base_prompt + gesture_modifiers.get(gesture, "") + gender_modifiers.get(gender, "")

        
        edges = canny(rgb_frame)
        edge_image = Image.fromarray(edges)

        
        result = pipe(
            prompt=dynamic_prompt,
            negative_prompt=negative_prompt,
            image=edge_image,
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=5,  
        ).images[0]

        
        result_np = np.array(result)
        preview = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        preview_upscaled = cv2.resize(preview, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

        
        frame_upscaled = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

       
        if preview_upscaled.shape[2] == 3 and frame_upscaled.shape[2] == 3:
            overlayed = cv2.addWeighted(frame_upscaled, 0.6, preview_upscaled, 0.4, 0)
        else:
            overlayed = preview_upscaled

        cv2.imshow(" Live AI Art Overlay", overlayed)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
