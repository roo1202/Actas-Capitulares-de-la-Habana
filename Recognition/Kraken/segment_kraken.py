import os
from PIL import Image
from kraken import binarization, pageseg
from kraken.lib import models

def is_image_binarized(image):
    """
    Verifica si la imagen ya está binarizada.
    La imagen binarizada tiene solo dos valores de color (blanco y negro).
    """
    # Convertimos a un arreglo numpy y verificamos los valores únicos
    img_array = image.convert("L")
    unique_values = set(img_array.getdata())
    return len(unique_values) <= 2  # Si tiene más de 2 valores, no está binarizada

def segment_image(image_path, seg_model_path, output_folder="lines_output"):
    """
    1) Carga el modelo de segmentación y la imagen.
    2) Binariza la imagen (si no está binarizada).
    3) Segmenta en líneas y guarda cada línea en una carpeta.
    Retorna la cantidad de líneas segmentadas.
    """

    # -- Validaciones de archivos --
    if not os.path.isfile(seg_model_path):
        raise FileNotFoundError(f"No se encontró el modelo de segmentación: {seg_model_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    # -- Cargar el modelo de segmentación --
    print("[INFO] Cargando modelo de segmentación...")
    # Note que no necesitamos cargar el modelo explícitamente para pageseg.segment()
    
    # -- Cargar la imagen --
    pil_img = Image.open(image_path).convert('L')

    # -- Verificar si la imagen está binarizada --
    if not is_image_binarized(pil_img):
        print("[INFO] La imagen no está binarizada. Realizando binarización...")
        bin_img = binarization.nlbin(pil_img)  # Otsu + deskew + limpieza interna por defecto
    else:
        print("[INFO] La imagen ya está binarizada.")
        bin_img = pil_img

    # -- Realizar la segmentación en líneas --
    seg_result = pageseg.segment(
        im=bin_img,
        text_direction='horizontal-lr',
        scale=1.0,
        maxcolseps=0  # Ajusta según sea necesario
    )

    if not hasattr(seg_result, 'boxes') or not seg_result.boxes:
        print("[WARNING] No se encontraron líneas. Revisa la calidad de la imagen y/o el modelo.")
        return 0

    # -- Crear carpeta de salida si no existe --
    os.makedirs(output_folder, exist_ok=True)

    # -- Recortar y guardar cada línea --
    for i, box in enumerate(seg_result['boxes'], start=1):
        line_crop = bin_img.crop(box)
        line_path = os.path.join(output_folder, f"line_{i:03d}.png")
        line_crop.save(line_path)

    print(f"[INFO] Segmentación completa: {len(seg_result.boxes)} líneas encontradas.")
    print(f"[INFO] Las líneas se han guardado en la carpeta: '{output_folder}'.")
    return len(seg_result.boxes)

if __name__ == "__main__":
    # Ejemplo de uso
    IMAGE_PATH = "binarized.jpg"
    SEG_MODEL_PATH = "sinai_sam_rec_v4_best.mlmodel"  # Modelo para segmentar
    OUTPUT_FOLDER = "lines_output"

    try:
        num_lines = segment_image(IMAGE_PATH, SEG_MODEL_PATH, OUTPUT_FOLDER)
        print(f"Total de líneas segmentadas: {num_lines}")
    except Exception as e:
        print("Ocurrió un error en la segmentación:", e)
