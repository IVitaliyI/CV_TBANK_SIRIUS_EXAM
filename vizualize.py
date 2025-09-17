import os
import cv2
import matplotlib.pyplot as plt

def draw_yolo_bboxes(image_path, label_path, class_names=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id, x_c, y_c, bw, bh = map(float, parts)
                cls_id = int(cls_id)

                x_c *= w
                y_c *= h
                bw *= w
                bh *= h

                x1 = int(x_c - bw / 2)
                y1 = int(y_c - bh / 2)
                x2 = int(x_c + bw / 2)
                y2 = int(y_c + bh / 2)

                # рисуем bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                label = str(cls_id) if class_names is None else class_names[cls_id]
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img


def visualize_folder(images_dir, labels_dir, class_names=None):
    images = sorted([f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))])
    index = [0]  # используем список, чтобы был mutable в обработчике событий

    plt.ion()
    fig, ax = plt.subplots(figsize=(8,8))
    img = draw_yolo_bboxes(os.path.join(images_dir, images[index[0]]),
                           os.path.join(labels_dir, os.path.splitext(images[index[0]])[0] + ".txt"),
                           class_names)
    im_display = ax.imshow(img)
    ax.axis("off")
    ax.set_title(images[index[0]])

    def on_key(event):
        if event.key == 'n':
            index[0] = (index[0] + 1) % len(images)
        elif event.key == 'p':
            index[0] = (index[0] - 1) % len(images)
        else:
            return

        img = draw_yolo_bboxes(os.path.join(images_dir, images[index[0]]),
                               os.path.join(labels_dir, os.path.splitext(images[index[0]])[0] + ".txt"),
                               class_names)
        im_display.set_data(img)
        ax.set_title(images[index[0]])
        fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)  # окно остаётся открытым, пока пользователь не закроет
    plt.ioff()


if __name__ == "__main__":
    images_dir = r"data/FINAL_DATASET/images/train"
    labels_dir = r"data/FINAL_DATASET/labels/train"
    class_names = ["tbank_logo"]

    visualize_folder(images_dir, labels_dir, class_names)