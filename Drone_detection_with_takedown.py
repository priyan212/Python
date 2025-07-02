import cv2, serial, time, threading, queue, numpy as np, torch
from ultralytics import YOLO

# CONFIG
SERIAL_PORT = "COM17"
BAUD_RATE = 115200
CAM_ID = 1
FRAME_W, FRAME_H = 640, 480
ALLOWED_NAMES = {"person"}  # Track only these
PAN_RANGE = (10, 170)
TILT_RANGE = (20, 160)
SMOOTHING = 0.18
LOCK_LOST_FRAMES = 20
DETECT_EVERY_N = 1
HUD_COLOR = (0, 255, 0)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
time.sleep(2)

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")
model.fuse()
if device == 0:
    model.to(device).half()
names = model.names

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_q = queue.Queue(maxsize=1)
def cam_worker():
    while True:
        ok, frm = cap.read()
        if not ok:
            continue
        if frame_q.full():
            try: frame_q.get_nowait()
            except queue.Empty: pass
        frame_q.put(frm)
threading.Thread(target=cam_worker, daemon=True).start()

def px_to_angle(px, span, rng):
    return rng[0] + (rng[1]-rng[0]) * np.clip(px/span, 0, 1)

last_sent = (-999, -999)
def send_angles(pan, tilt):
    global last_sent
    p, t = int(pan), int(tilt)
    if (p, t) != last_sent:
        ser.write(f"P{p:03d} T{t:03d}\n".encode())
        last_sent = (p, t)

# Tracking state
cur_pan = np.mean(PAN_RANGE)
cur_tilt = np.mean(TILT_RANGE)
lock_id, lost_counter = None, 0
takedown_armed = False
button_pressed = False

def draw_hud(img, fps, pan, tilt, locked, sweep_phase):
    global takedown_armed

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    overlay = img.copy()
    cv2.line(overlay, (cx, 0), (cx, h), HUD_COLOR, 1)
    cv2.line(overlay, (0, cy), (w, cy), HUD_COLOR, 1)

    # Sweep animation
    angle = (sweep_phase % 360) * np.pi/180
    x2 = int(cx + 200*np.cos(angle))
    y2 = int(cy + 200*np.sin(angle))
    cv2.line(overlay, (cx, cy), (x2, y2), HUD_COLOR, 1)

    # Panel
    panel = np.zeros((85, 200, 3), dtype=np.uint8)
    cv2.putText(panel, f"FPS   : {fps:5.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(panel, f"PAN   : {int(pan):3d}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(panel, f"TILT  : {int(tilt):3d}",(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    status_txt = "LOCKED" if locked else "SEARCH"
    color_txt  = (0,0,255) if locked else (0,255,255)
    cv2.putText(panel, status_txt, (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_txt, 2)

    alpha = 0.25
    img[:] = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    img[10:95, 10:210] = cv2.addWeighted(img[10:95, 10:210], 0.3, panel, 0.7, 0)

    # TAKEDOWN button
    btn_color = (0, 255, 0) if not takedown_armed else (0, 0, 255)
    x, y, w, h = 480, 20, 180, 45
    cv2.rectangle(img, (x, y), (x + w, y + h), btn_color, -1)
    cv2.putText(img, "TAKEDOWN", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

def mouse_callback(event, mx, my, flags, param):
    global button_pressed
    if event == cv2.EVENT_LBUTTONDOWN and takedown_armed:
        x, y, w, h = 480, 20, 180, 45
        if x <= mx <= x + w and y <= my <= y + h:
            print("TAKEDOWN CLICKED")
            button_pressed = True

cv2.namedWindow("Defence-HUD Tracker")
cv2.setMouseCallback("Defence-HUD Tracker", mouse_callback)

# Main loop
prev_t = time.time()
frame_id = 0
sweep = 0

while True:
    try:
        frame = frame_q.get(timeout=1)
    except queue.Empty:
        continue

    frame_id += 1
    run_detect = (frame_id % DETECT_EVERY_N == 0)
    target = None

    if run_detect:
        with torch.cuda.amp.autocast(enabled=device==0):
            res = model.track(frame, conf=0.3, iou=0.45, verbose=False, persist=True, device=device)[0]

        valid = []
        for b in res.boxes:
            if b.id is None: continue
            label = names[int(b.cls)]
            if label not in ALLOWED_NAMES: continue
            x1, y1, x2, y2 = b.xyxy.cpu().numpy()[0]
            valid.append({"id": int(b.id), "box": (x1,y1,x2,y2), "area": (x2-x1)*(y2-y1), "name": label})
        if lock_id is not None:
            match = next((v for v in valid if v["id"] == lock_id), None)
            if match:
                lost_counter = 0
                target = match
            else:
                lost_counter += 1
                if lost_counter >= LOCK_LOST_FRAMES:
                    lock_id = None
        if lock_id is None and valid:
            target = max(valid, key=lambda v: v["area"])
            lock_id, lost_counter = target["id"], 0

    if target:
        x1, y1, x2, y2 = target["box"]
        cx_t, cy_t = (x1 + x2)/2, (y1 + y2)/2
        target_pan = px_to_angle(FRAME_W - cx_t, FRAME_W, PAN_RANGE)
        target_tilt = px_to_angle(FRAME_H - cy_t, FRAME_H, TILT_RANGE)
        cur_pan = cur_pan * (1 - SMOOTHING) + target_pan * SMOOTHING
        cur_tilt = cur_tilt * (1 - SMOOTHING) + target_tilt * SMOOTHING
        send_angles(cur_pan, cur_tilt)

        # Only arm takedown if target is centered
        dx = abs(cx_t - FRAME_W / 2)
        dy = abs(cy_t - FRAME_H / 2)
        takedown_armed = dx < 80 and dy < 80

        cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        cv2.line(frame, (int(cx_t-10),int(cy_t)), (int(cx_t+10),int(cy_t)), (0,0,255), 1)
        cv2.line(frame, (int(cx_t),int(cy_t-10)), (int(cx_t),int(cy_t+10)), (0,0,255), 1)
        cv2.putText(frame, target["name"], (int(x1), int(y1)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        takedown_armed = False

    if button_pressed:
        print("Sending FIRE command to Arduino")
        ser.write(b"FIRE\n")
        button_pressed = False

    now = time.time()
    fps = 1.0 / (now - prev_t) if now != prev_t else 0
    prev_t = now
    sweep = (sweep + 4) % 360
    draw_hud(frame, fps, cur_pan, cur_tilt, target is not None, sweep)

    cv2.imshow("Defence-HUD Tracker", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
ser.close()
cv2.destroyAllWindows()
