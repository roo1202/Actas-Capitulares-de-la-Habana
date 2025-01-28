import os
from PIL import Image
from kraken import binarization, pageseg, rpred
from kraken.lib import models
def is_image_binarized(image):
    """
    Verifica si la imagen ya está binarizada.
    La imagen binarizada tiene solo dos valores de color (blanco y negro).
    """
    img_array = image.convert("L")
    unique_values = set(img_array.getdata())
    return len(unique_values) <= 2  # Si tiene más de 2 valores, no está binarizada

def segment_image(image_path, output_folder="lines_output", text_direction="horizontal-lr"):
    """
    1) Carga la imagen y verifica su binarización.
    2) Segmenta en líneas y guarda cada línea en una carpeta.
    Retorna la cantidad de líneas segmentadas.
    """

    # -- Validaciones de archivos --
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    # -- Cargar la imagen --
    pil_img = Image.open(image_path).convert('L')

    # -- Verificar si la imagen está binarizada --
    if not is_image_binarized(pil_img):
        print("[INFO] La imagen no está binarizada. Realizando binarización...")
        bin_img = binarization.nlbin(im=pil_img)  # Otsu + deskew + limpieza interna por defecto
    else:
        print("[INFO] La imagen ya está binarizada.")
        bin_img = pil_img

    # -- Realizar la segmentación en líneas --
    seg_result = pageseg.segment(
        im=bin_img,
        text_direction=text_direction,
        scale=None,  # Estima automáticamente la escala si es None
        maxcolseps=0,  # Ajustar según el tipo de documento
        no_hlines=False  # Si no hay líneas horizontales, activarlo
    )

    if not hasattr(seg_result, 'boxes') or not seg_result.boxes:
        print("[WARNING] No se encontraron líneas. Revisa la calidad de la imagen.")
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

def recognize_lines_from_segmented_image(lines_folder, rec_model_path):
    """
    1) Carga el modelo de reconocimiento y las líneas segmentadas.
    2) Realiza el reconocimiento de cada línea usando `rpred()`.
    3) Devuelve el texto reconocido de todas las líneas.
    """

    # -- Validar carpeta de líneas y modelo --
    if not os.path.isdir(lines_folder):
        raise FileNotFoundError(f"La carpeta de líneas no existe: {lines_folder}")
    if not os.path.isfile(rec_model_path):
        raise FileNotFoundError(f"No se encontró el modelo de reconocimiento: {rec_model_path}")

    # -- Cargar el modelo de reconocimiento --
    print("[INFO] Cargando modelo de reconocimiento...")
    rec_nn = models.load_any(rec_model_path)

    # -- Obtener lista de archivos de línea en la carpeta --
    line_files = sorted(
        f for f in os.listdir(lines_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
    )
    if not line_files:
        print("[WARNING] No hay imágenes de línea en la carpeta especificada.")
        return ""

    # -- Cargar cada imagen de línea en memoria --
    lines_crops = []
    for line_file in line_files:
        line_path = os.path.join(lines_folder, line_file)
        pil_line = Image.open(line_path).convert('L')
        lines_crops.append(pil_line)

    # -- Reconocer texto de cada línea usando rpred() --
    print("[INFO] Iniciando reconocimiento de texto...")
    seg_result = pageseg.segment(im=lines_crops[0], text_direction='horizontal-lr')
    recognized_text = ""
    for line_crop in lines_crops:
        preds = rpred.rpred(rec_nn, line_crop, seg_result)
        recognized_text += "\n".join([pred.outputs[0].strip() for pred in preds]) + "\n"

    return recognized_text

if __name__ == "__main__":
    # Ejemplo de uso
    IMAGE_PATH = "output.png"
    REC_MODEL_PATH = "McCATMuS_nfd_nofix_V1.mlmodel"  # Modelo para reconocer
    OUTPUT_FOLDER = "data"

    try:
        num_lines = segment_image(IMAGE_PATH, OUTPUT_FOLDER)
        print(f"Total de líneas segmentadas: {num_lines}")
        recognized_text = recognize_lines_from_segmented_image(OUTPUT_FOLDER, REC_MODEL_PATH)
        print("Texto reconocido:\n")
        print(recognized_text)
    except Exception as e:
        print("Ocurrió un error:", e)
