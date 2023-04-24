import cv2
from tensorflow import keras
import numpy as np


def predict(image_path):
    # Danh sách biển báo nhận dạng đươợc
    traffice_result = []

    #Load model
    model = keras.models.load_model("trafficsign.h5", compile=False)

    # Nhãn dự đoán
    categories = ['Không phải biển báo', 'Biển báo cấm', 'Biển báo chỉ dẫn', 'Biển báo hiệu lệnh', 'Biển báo nguy hiểm']

    # Tải ảnh đầu vào
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img_array = np.array(img, dtype="float") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_batch)

    # Sắp xếp các giá trị độ chính xác theo thứ tự giảm dần
    sorted_indices = np.argsort(pred)[0][::-1]

    # In ra các nhãn tương ứng và giá trị độ chính xác của chúng
    for index in sorted_indices:
        label = categories[index]
        accuracy = pred[0][index]

        traffice_result.append({
            "id": str(index),
            "name": str(label),
            "accuracy": str(round(accuracy*100, 2)) + "%"
        })

    return traffice_result