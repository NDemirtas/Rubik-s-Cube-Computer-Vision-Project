import cv2

CAMERA_INDEX = 0          # change to 1, 2 ... for another camera

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise SystemExit(f"Could not open camera {CAMERA_INDEX}")

cv2.namedWindow("Live Video Feed", cv2.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Live Video Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):   # Q or Esc to quit
        break

cap.release()
cv2.destroyAllWindows()