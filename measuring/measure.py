import cv2
import numpy as np
import os 

#modes = all (all tumor px), largest (only largest part in px), parts (if more parts of tumor -> [part1: px_value1, part2: px_value2], ...)
def measure(mask_path:str , pixel_spacing:tuple[float,float] = False, mode: str = "parts"):
    if not os.path.exists(mask_path):
        return "invalid path or image"
    
    if not pixel_spacing or pixel_spacing[0] != pixel_spacing[1]:
        return ("invalid pixel spacing argument")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_2d = (mask > 0).astype(np.uint8)

    area_px = int(mask_2d.sum()) #area of tumor in pixels
    
    if area_px == 0:
        return "no tumor found"
    
    spacing_x, spacing_y = pixel_spacing
    pixel_area_mm2 = spacing_x * spacing_y
    
    area_mm2 = round((area_px * pixel_area_mm2), 2) # -\\- in mm2

    if mode == "all":
        return f"{area_mm2}mm^2"

    num,labels,stats,_ = cv2.connectedComponentsWithStats(mask_2d, connectivity=8)
    components_stats = [stats[i] for i in range(1,num)]
    #example component : array([ 267,  176,  113,   77, 4569], dtype=int32) -> left,top,width,height,area_px
    tumors_area_px = [component[-1] for component in components_stats]

    tumors_parts_mm2 = [round(float((tumor * pixel_area_mm2)),2) for tumor in tumors_area_px]

    if mode == "largest":
        return f"{max(tumors_parts_mm2)} mm^2"

    if mode == "parts":
        tumors = {}
        for tumor_mm2 in tumors_parts_mm2:
            tumors[f"tumor{len(tumors) + 1}"] = f"{tumor_mm2} mm^2"
        return tumors
    
    return f"invalid action: {mode}"

# print(measure("../segmentation/dummy_mask.png", (0.49,0.49), "parts"))

