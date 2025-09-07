import os
import cv2
from tqdm import tqdm
import shutil

# IMPORTANT: Apne folder structure ke hisab se in paths ko aache se check karein
# source_data_dir woh folder hai jahan aapne VisDrone dataset rakha hai
source_data_dir = 'datasets' 
# target_dir woh folder hai jahan naye YOLO format ke labels aur images save honge
target_dir = 'VisDrone-YOLO-format'

# VisDrone dataset ke class IDs aur names.
# Hum 'ignored-regions' (0) aur 'others' (11) ko skip kar denge.
# YOLO ke liye class IDs 0 se shuru honi chahiye.
# Isliye VisDrone class '1' (pedestrian) hamare liye class '0' hogi.
visdrone_classes = {
    '1': 0,  # pedestrian -> 0
    '2': 1,  # person -> 1
    '3': 2,  # car -> 2
    '4': 3,  # van -> 3
    '5': 4,  # bus -> 4
    '6': 5,  # truck -> 5
    '7': 6,  # motor -> 6
    '8': 7,  # bicycle -> 7
    '9': 8,  # awning-tricycle -> 8
    '10': 9, # tricycle -> 9
}

def convert_visdrone_to_yolo(data_split):
    """
    VisDrone annotations ko YOLO format mein convert karta hai.
    data_split: 'train', 'val', ya 'test' ho sakta hai.
    """
    print(f"Processing '{data_split}' data...")
    
    # Source paths
    source_img_dir = os.path.join(source_data_dir, f'VisDrone2019-DET-{data_split}', 'images')
    source_ann_dir = os.path.join(source_data_dir, f'VisDrone2019-DET-{data_split}', 'annotations')

    # Target paths
    target_img_dir = os.path.join(target_dir, 'images', data_split)
    target_lbl_dir = os.path.join(target_dir, 'labels', data_split)

    # Zaroori folders banayein agar woh maujood nahi hain
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)

    # Sabhi annotation files par loop karein
    annotation_files = os.listdir(source_ann_dir)
    for filename in tqdm(annotation_files, desc=f"Converting {data_split} annotations"):
        # Image path
        img_path = os.path.join(source_img_dir, filename.replace('.txt', '.jpg'))
        
        # Original image ko target folder mein copy karein
        # shutil.copy(img_path, target_img_dir) -> Agar aap images bhi copy karna chahte hain
        
        # Image ki height aur width padhein
        try:
            shutil.copy(img_path, target_img_dir)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        # Annotation file padhein
        with open(os.path.join(source_ann_dir, filename), 'r') as f:
            lines = f.readlines()

        yolo_annotations = []
        for line in lines:
            parts = line.strip().split(',')
            # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            
            bbox_left = float(parts[0])
            bbox_top = float(parts[1])
            bbox_w = float(parts[2])
            bbox_h = float(parts[3])
            class_id = parts[5]

            # Ignored regions aur others ko skip karein
            if class_id in visdrone_classes:
                yolo_class_id = visdrone_classes[class_id]
                
                # Coordinates ko normalize karein
                center_x = (bbox_left + bbox_w / 2) / img_w
                center_y = (bbox_top + bbox_h / 2) / img_h
                norm_w = bbox_w / img_w
                norm_h = bbox_h / img_h
                
                yolo_annotations.append(f"{yolo_class_id} {center_x} {center_y} {norm_w} {norm_h}")

        # Nayi YOLO annotation file likhein
        with open(os.path.join(target_lbl_dir, filename), 'w') as f:
            f.write('\n'.join(yolo_annotations))

# if __name__ == '__main__':
#     # Train aur validation data dono ke liye function call karein
#     convert_visdrone_to_yolo('train')
#     convert_visdrone_to_yolo('val')
#     # Test set ke liye bhi convert kar sakte hain, agar zaroorat ho
#     # convert_visdrone_to_yolo('test-dev')

#     print("Conversion complete!")
#     print(f"YOLO formatted data is saved in '{target_dir}' folder.")
if __name__ == '__main__':
    # Train aur validation data dono ke liye function call karein
    convert_visdrone_to_yolo('train')
    convert_visdrone_to_yolo('val')

    print("Conversion complete!")
    print(f"YOLO formatted data is saved in '{target_dir}' folder.")