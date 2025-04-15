import time
import cv2
import numpy as np

def video_processing():
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)

    ref_image = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
    fly_image = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    variant_image = cv2.imread('variant-1.jpg', cv2.IMREAD_GRAYSCALE)

    if ref_image is None:
        print("Ошибка загрузки изображения метки.")
        return
    if fly_image is None:
        print("Ошибка загрузки изображения мухи.")
        return
    if variant_image is None:
        print("Ошибка загрузки изображения variant-1.jpg.")
        return

    variant_image_resized = cv2.resize(variant_image, (720, 540), interpolation=cv2.INTER_LINEAR)

    template_h, template_w = ref_image.shape[:2]
    fly_h, fly_w = fly_image.shape[:2]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата изображения.")
            break

        frame_resized = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        found = None
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = cv2.resize(gray_frame, (int(gray_frame.shape[1] * scale), int(gray_frame.shape[0] * scale)))
            r = gray_frame.shape[1] / float(resized.shape[1])
            if resized.shape[0] < template_h or resized.shape[1] < template_w:
                break
            result = cv2.matchTemplate(resized, ref_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if found is None or max_val > found[0]:
                found = (max_val, max_loc, r)

        if found and found[0] > 0.6:
            _, max_loc, r = found
            top_left = (int(max_loc[0] * r), int(max_loc[1] * r))
            bottom_right = (int((max_loc[0] + template_w) * r), int((max_loc[1] + template_h) * r))
            cv2.rectangle(frame_resized, top_left, bottom_right, (0, 255, 0), 2)

            center_x = (top_left[0] + bottom_right[0]) // 2
            center_y = (top_left[1] + bottom_right[1]) // 2

            coordinates_text = f"({center_x}, {center_y})"
            print(coordinates_text)

            fly_center_x = center_x - fly_w // 2
            fly_center_y = center_y - fly_h // 2

            for i in range(fly_h):
                for j in range(fly_w):
                    if fly_image[i, j, 3] > 0: 
                        frame_resized[fly_center_y + i, fly_center_x + j] = fly_image[i, j, :3]

            cv2.putText(frame_resized, coordinates_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            print("Метка не обнаружена")
            cv2.putText(frame_resized, "Метка не обнаружена", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Frame', frame_resized)
        cv2.imshow('Variant Image', variant_image_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
#main func
if __name__ == '__main__':
    video_processing()
