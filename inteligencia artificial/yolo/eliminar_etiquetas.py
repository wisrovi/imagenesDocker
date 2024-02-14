

# Lista de clases que deseas mantener
FILTER_CLASS = [0, 1, 2]


import os



# Ruta de la carpeta que contiene las carpetas train, test y valid
dataset_path = "./Car-Damage-V5-6/"
# dataset_path = "./damages-car2/"
# dataset_path = "./damages-car-3/"
# dataset_path = "./damages-car-4/"




# Función para filtrar las clases en un archivo de etiquetas
def filter_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filtrar las líneas que contienen las clases que deseas mantener
    filtered_lines = [line for line in lines if int(line.split()[0]) in FILTER_CLASS]

    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)

# Iterar a través de las carpetas train, test y valid
for split_folder in ['train', 'test', 'valid']:
    split_path = os.path.join(dataset_path, split_folder)

    # Iterar a través de las carpetas de etiquetas en cada conjunto
    for label_folder in os.listdir(os.path.join(split_path, 'labels')):
        label_folder_path = os.path.join(split_path, 'labels')
        print(label_folder_path)

        # Iterar a través de los archivos de etiquetas en cada carpeta
        for label_file in os.listdir(label_folder_path):
            label_file_path = os.path.join(label_folder_path, label_file)
            #print(label_file_path)

            # Filtrar las clases en el archivo de etiquetas
            filter_labels(label_file_path)

print("Proceso completado.")