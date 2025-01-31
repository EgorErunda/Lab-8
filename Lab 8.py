import cv2

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

        text_x = 10
        text_y = 30
        cv2.putText(resized_gray, f"Координаты метки: ({x}, {y})", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        resized_img = cv2.resize(img, new_size)
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.drawContours(resized_img, [largest_contour], -1, (0, 255, 0), 2)

        cv2.imshow('Изображение с меткой', resized_img)

if __name__ == '__main__':
    image_processing()

cv2.waitKey(0)
cv2.destroyAllWindows()
