import cv2
import numpy as np

def image_processing():
    img = cv2.imread('images/variant-1.jpg')
    if img is None:
        print("Ошибка загрузки изображения.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    max_width = 1200
    max_height = 1000
    height, width = img.shape[:2]
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)
    new_size = (int(width * scale), int(height * scale))
    resized_gray = cv2.resize(gray, new_size)

    ret, thresh = cv2.threshold(resized_gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        print(f"Координаты метки: ({x}, {y})")

        resized_img = cv2.resize(img, new_size)
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawContours(resized_img, [largest_contour], -1, (0, 255, 0), 2)

        fly_img = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
        fly_height, fly_width, _ = fly_img.shape
        fly_center_x = x + w // 2
        fly_center_y = y + h // 2
        fly_x = fly_center_x - fly_width // 2
        fly_y = fly_center_y - fly_height // 2
        roi = resized_img[fly_y:fly_y+fly_height, fly_x:fly_x+fly_width]
        roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(fly_img[:, :, 3]))
        roi_fg = cv2.bitwise_and(fly_img[:, :, 0:3], fly_img[:, :, 0:3], mask=fly_img[:, :, 3])
        resized_img[fly_y:fly_y+fly_height, fly_x:fly_x+fly_width] = cv2.add(roi_bg, roi_fg)

        cv2.imshow('Изображение с меткой', resized_img)
        cv2.imshow('Grayscale Image', resized_gray)

if __name__ == '__main__':
    image_processing()

cv2.waitKey(0)
cv2.destroyAllWindows()
