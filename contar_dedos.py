import cv2
import numpy as np

def draw_text_with_outline(img, text, org, font, font_scale, color, thickness, outline_color=(0,0,0), outline_thickness=3):
    cv2.putText(img, text, org, font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

def draw_info_panel(frame, lines, roi_rect=None, margin=12):
    """
    Panel semitransparente con varias líneas de texto, colocado fuera del ROI.
    lines: lista de strings.
    roi_rect: (x, y, w, h) del ROI para evitar superposición.
    """
    H, W = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.9
    th = 2
    line_gap = 10

    # Tamaño del panel según el texto
    text_sizes = [cv2.getTextSize(t, font, fs, th)[0] for t in lines]
    text_w = max((w for (w, h) in text_sizes), default=0)
    text_h = sum((h for (w, h) in text_sizes), 0) + line_gap * (max(len(lines) - 1, 0))

    pad_x, pad_y = 14, 14
    panel_w = text_w + 2 * pad_x
    panel_h = text_h + 2 * pad_y

    # Posición preferida y candidatas (UL, UR, BL, BR)
    x, y = margin, margin

    def overlaps(ax, ay, aw, ah, bx, by, bw, bh):
        return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)

    if roi_rect is not None:
        rx, ry, rw, rh = roi_rect
        candidates = [
            (margin, margin),
            (W - panel_w - margin, margin),
            (margin, H - panel_h - margin),
            (W - panel_w - margin, H - panel_h - margin),
        ]
        # Elige la primera que no solape con el ROI
        for cx, cy in candidates:
            if not overlaps(cx, cy, panel_w, panel_h, rx, ry, rw, rh):
                x, y = cx, cy
                break
        else:
            # Si todas solapan, ajusta para que al menos no tape completamente
            x = max(margin, min(x, W - panel_w - margin))
            y = max(margin, min(y, H - panel_h - margin))

    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (30, 30, 30), -1)  # gris oscuro
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Texto blanco con contorno negro
    ty = y + pad_y
    for t in lines:
        sz = cv2.getTextSize(t, font, fs, th)[0]
        draw_text_with_outline(frame, t, (x + pad_x, ty + sz[1]), font, fs, (255, 255, 255), th)
        ty += sz[1] + line_gap


# ------------------ DETECCIÓN DE DEDOS ------------------
def count_fingers(contour, roi_frame):
    """
    Cuenta los dedos levantados a partir del contorno,
    usando el centroide para filtrar defectos falsos.

    Devuelve: (conteo_defectos, solidez)
    """
    hull_points = cv2.convexHull(contour)
    if hull_points is None:
        return 0, 0

    hull_area = cv2.contourArea(hull_points)

    M = cv2.moments(contour)
    if M['m00'] == 0 or hull_area == 0:
        return 0, 0

    contour_area = M['m00']
    solidity = float(contour_area) / hull_area

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return 0, solidity

    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return 0, solidity

    finger_count = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))

        if b * c == 0:
            continue

        angle = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))

        # Puntos entre dedos: ángulo agudo, profundidad suficiente, y por encima del centroide + margen
        if angle <= 90 and d > 7000 and far[1] < (cy + 25):
            finger_count += 1
            cv2.circle(roi_frame, far, 5, (0, 0, 255), -1)

    return finger_count, solidity


# main
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede acceder a la cámara.")
        return

    print("Presiona 'q' para salir.")

    roi_w = 250
    roi_h = 300

    stable_target = None
    stable_frames = 0
    STABLE_N = 60
    gesture_confirmed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        frame_copy = frame.copy()

        # ROI centrado
        roi_x = max((frame_w - roi_w) // 2, 0)
        roi_y = max((frame_h - roi_h) // 2, 0)

        if roi_x + roi_w > frame_w:
            roi_x = frame_w - roi_w
        if roi_y + roi_h > frame_h:
            roi_y = frame_h - roi_h

        box_color = (0, 255, 0) if gesture_confirmed else (255, 0, 0)
        cv2.rectangle(frame_copy, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), box_color, 2)

        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Segmentación de piel en YCrCb
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], np.uint8)
        upper = np.array([255, 173, 127], np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display_count = None
        solidity = 0.0

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)

            if area > 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                x_global = x + roi_x
                y_global = y + roi_y

                ratio = h / float(w) if w > 0 else 0

                if 0.4 < ratio < 4.0:
                    cv2.rectangle(frame_copy, (x_global, y_global), (x_global + w, y_global + h), (0, 255, 0), 2)

                    num_defects, solidity = count_fingers(cnt, roi)

                    if solidity > 0.85:
                        display_count = 0
                    else:
                        display_count = num_defects + 1

        # --- Estabilidad por 60 frames ---
        if display_count is not None:
            if stable_target is None or display_count != stable_target:
                stable_target = display_count
                stable_frames = 1
                gesture_confirmed = False
            else:
                stable_frames += 1
                if stable_frames >= STABLE_N:
                    gesture_confirmed = True
        else:
            stable_target = None
            stable_frames = 0
            gesture_confirmed = False

        # Interfaz (arriba izq)
        panel_lines = []
        if display_count is not None:
            panel_lines.append(f"Dedos: {display_count}")
        panel_lines.append(f"Solidez: {solidity:.2f}")
        panel_lines.append(f"Estable: {min(stable_frames, STABLE_N)}/{STABLE_N}")
        if gesture_confirmed and stable_target is not None:
            panel_lines.append("Gesto confirmado")

        roi_rect = (roi_x, roi_y, roi_w, roi_h)
        draw_info_panel(frame_copy, panel_lines, roi_rect=roi_rect)

        # Ventanas
        cv2.imshow("Mascara piel (ROI)", mask)
        cv2.imshow("Deteccion de dedos", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
