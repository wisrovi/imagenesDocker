import os
import shutil

# Rutas de las carpetas de train y val
val_images_dir = "val/images/"

# Obtener la lista de archivos en la carpeta de validación
val_images = os.listdir(val_images_dir)

# Calcular la cantidad de archivos a mover
num_files_to_move = int(len(val_images) * 0.80)

# Mover la mitad de los archivos de imágenes y etiquetas de val a train
for i in range(num_files_to_move):
    # Mover imágenes
    image_filename = val_images[i]
    origen_image = val_images_dir + image_filename
    destino_image = "train/images/" + image_filename

    # Mover etiquetas
    val_label = str(destino_image).replace("images", "labels")
    extension = val_label[-3:]
    destino_label = val_label.replace(extension, "txt")

    origen_label = destino_label.replace("train", "val")

    # validar que los origen existan
    if os.path.exists(origen_image) and os.path.exists(origen_label):
        shutil.move(origen_label, destino_label)
        shutil.move(origen_image, destino_image)
        print("Se movió el archivo", origen_image, "a", destino_image)
        print("Se movió el archivo", origen_label, "a", destino_label)

    print("\t", os.path.exists(origen_image), origen_image, destino_image)
    print("\t", os.path.exists(origen_label), origen_label, destino_label)
    print()

print("Se han movido los archivos de val a train.")
