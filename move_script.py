import os, shutil, random

# пути
images_dir = r"data/annotaded"
labels_dir = r"data/labels_my-project-name_2025-09-17-11-19-32"

train_img_dir = r"data/YOLO_dataset/images/train"
val_img_dir = r"data/YOLO_dataset/images/val"
train_lbl_dir = r"data/YOLO_dataset/labels/train"
val_lbl_dir = r"data/YOLO_dataset/labels/val"

# создаём папки
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# список файлов
images = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
random.shuffle(images)

split_idx = int(0.8 * len(images))  # 80% train
train_files = images[:split_idx]
val_files = images[split_idx:]

COUNT = 0

def move_files(files, img_dest, lbl_dest):
    global COUNT
    for img in files:
        img_name = os.path.splitext(img)[0]
        lbl = img_name + ".txt"

        # копируем картинку
        shutil.copy(os.path.join(images_dir, img), os.path.join(img_dest, img))

        # копируем или создаём пустую разметку
        lbl_src = os.path.join(labels_dir, lbl)
        lbl_dst = os.path.join(lbl_dest, lbl)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)
            COUNT += 1
        else:
            open(lbl_dst, "w").close()  # пустой файл
    
move_files(train_files, train_img_dir, train_lbl_dir)
move_files(val_files, val_img_dir, val_lbl_dir)
print(f"Скопированно {COUNT} картинок")


