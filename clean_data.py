import os
import cv2
import imutils
from tqdm import tqdm

def crop_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    return img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    training = os.path.join(BASE_DIR, "data", "raw", "Training")
    testing = os.path.join(BASE_DIR, "data", "raw", "Testing")
    IMG_SIZE = 256

    print("📁 Training path:", training)
    print("📁 Testing path:", testing)

    for dataset, name in [(training, "Training"), (testing, "Testing")]:
        print(f"\n🚀 Xử lý tập {name}...")
        for dir in os.listdir(dataset):
            print(f"🔹 {name}/{dir}")
            path = os.path.join(dataset, dir)
            save_path = os.path.join(BASE_DIR, "data", "cleaned", name, dir)
            os.makedirs(save_path, exist_ok=True)

            for img in tqdm(os.listdir(path), desc=f"{name}/{dir}", ncols=80):
                image = cv2.imread(os.path.join(path, img))
                if image is None:
                    print(f"⚠️ Lỗi đọc ảnh: {img}")
                    continue
                new_img = cv2.resize(crop_img(image), (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(save_path, img), new_img)

    print("\n✅ Xong! Dữ liệu đã được lưu tại:", os.path.join(BASE_DIR, "data", "cleaned"))