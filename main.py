import cv2
import mediapipe as mp
import serial
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================
# ARDUINO
# ==========================

arduino = serial.Serial("COM5", 9600)
time.sleep(2)

# ==========================
# MEDIAPIPE
# ==========================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ==========================
# CAMARA
# ==========================

cam = cv2.VideoCapture(0)

ultimo_valido = 0
contador_estable = 0
ultimo_envio = 0

while True:

    ok, frame = cam.read()

    if not ok:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        dedos_detectados = 0

        # Índice
        if hand[8].y < hand[6].y:
            dedos_detectados += 1

        # Medio
        if hand[12].y < hand[10].y:
            dedos_detectados += 1

        # Anular
        if hand[16].y < hand[14].y:
            dedos_detectados += 1

        # Meñique
        if hand[20].y < hand[18].y:
            dedos_detectados += 1

        # Pulgar
        if hand[4].x < hand[3].x:
            dedos_detectados += 1

        # Si detecta al menos 1 dedo, guardar valor
        if dedos_detectados > 0:

            if dedos_detectados == ultimo_valido:
                contador_estable += 1
            else:
                contador_estable = 0

            ultimo_valido = dedos_detectados
        else:
            contador_estable = 0
            ultimo_valido = 0

        # Dibujar landmarks
        h, w, _ = frame.shape

        for point in hand:

            x = int(point.x * w)
            y = int(point.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

    # Reenviar continuamente el último valor válido
    if contador_estable >= 5:

        ahora = time.time()

        if ahora - ultimo_envio >= 0.2:

            print(f"Enviando: {ultimo_valido}")

            arduino.write(
                f"{ultimo_valido}\n".encode()
            )

            ultimo_envio = ahora
    else:
        print(f"Enviando: {ultimo_valido}")

    cv2.putText(
        frame,
        f"Dedos: {ultimo_valido}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Estable: {contador_estable}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "Detector de Mano",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
arduino.close()