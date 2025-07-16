import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not detected.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not captured.")
        break

    cv2.imshow("🧪 Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
