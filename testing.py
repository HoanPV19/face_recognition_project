import face_recognition
import cv2

# Load ảnh mẫu
known_image = face_recognition.load_image_file("C:/Users/hoany/Desktop/Python/face_recognition_project/face_test/day.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Khởi tạo video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Đọc một frame từ webcam
    ret, frame = video_capture.read()

    # Tìm các khuôn mặt trong frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # So sánh encoding của khuôn mặt trong frame với encoding của ảnh mẫu
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        name = "Unknown"
        confidence = 0  # Biến lưu trữ phần trăm khớp
        if True in matches:
            # Tính toán khoảng cách cosine (giới hạn 0 - 1, 0 là giống hoàn toàn)
            face_distances = face_recognition.face_distance([known_encoding], face_encoding)
            confidence = (1 - face_distances[0]) * 100  # Chuyển thành phần trăm

            # Chọn ngưỡng (tùy chỉnh)
            if confidence >= 60:  # Giả sử 60% là đủ nhận diện
                name = "Known Face ({:.1f}%)".format(confidence)
            else:
                name = "Maybe Known ({:.1f}%)".format(confidence)

        # Vẽ hình chữ nhật quanh khuôn mặt và ghi tên
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Hiển thị frame
    cv2.imshow('Video', frame)

    # Nếu nhấn 'q' thì thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng các tài nguyên
video_capture.release()
cv2.destroyAllWindows()
