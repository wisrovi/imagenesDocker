import os
import json
import shutil
import pandas as pd
from PIL import Image

class_mapper = {
    '1': 0,
    '2': 1,
    '3': -1,
    '4': 1,
    '5': 0,
    '6': -1,
    '7': -1,
    '8': -1,
    '9': -1,
    '11': -1
}

def list_dir_folders(path):

    folders = []

    for element in os.listdir(path):
        element_path = os.path.join(path, element)

        if os.path.isdir(element_path):
            folders.append(os.path.join(path,element))
        
    return folders

def get_json(path):

    for element in os.listdir(path):
        if element.endswith('json'):
            return element

def write_annotation(file, row):

    mask = [None]*(len(row['x'])+len(row['y']))

    mask[::2] = row['x']
    mask[1::2] = row['y']

    with open(file, 'a+') as f:

        f.write(f'{row["class"]} {" ".join([str(coord) for coord in mask])}\n')



"""
1 - Capó
2 - Marco
3 - Faro
4 - Paragolpes delantero
44- Paragolpes Trasero
5 - Puerta
6 - Cristal
7 - Luna
8 - Techo
9 - Maletero
10 - Retrovisor
11- Matricula	
12- Ruedas
13- Aleta	
14- Faldon
15- Marco+Aleta
"""



TAGS = [
2, 
3,
4,
44,
1,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15
]








if __name__ == '__main__':

    TRAIN_SPLIT = .8

    annotations = []

    os.makedirs('processed_dataset/train/images', exist_ok=True)
    os.makedirs('processed_dataset/train/labels', exist_ok=True)
    os.makedirs('processed_dataset/val/images', exist_ok=True)
    os.makedirs('processed_dataset/val/labels', exist_ok=True)

    # if True:
    folder__ =  "/home/willians/Downloads/Recursos/seg/Partes/funcionales_20230921/"
    for folder in list_dir_folders(folder__):
        json_file = get_json(folder)
        
        print(folder, json_file)

        with open(os.path.join(folder, json_file), 'r') as f:
            this_folder_annotations = json.load(f)
        
        for _, annotation in this_folder_annotations.items():

            filename = os.path.join(folder, annotation['filename'])

            try:
            
                img = Image.open(filename)

                width = img.width
                height = img.height

                for region in annotation['regions']:

                    x = [coord/width for coord in region['shape_attributes']['all_points_x']]
                    y = [coord/height for coord in region['shape_attributes']['all_points_y']]

                    _class = region['region_attributes']['name']
                    
                    if _class is not None and _class != -1:
                        annotations.append({
                            'filename': filename,
                            'class': TAGS.index(int(_class)),  # int(_class)-1,
                            'x': x,
                            'y': y
                        })
            except:
                pass

    annotations = pd.DataFrame(annotations)

    print(f'Etiquetas: {len(annotations)}')

    print(annotations['class'].value_counts())






    # max_num_instances = annotations['class'].value_counts().min()

    # annotations = pd.concat([subdf[-1].sample(max_num_instances, random_state=42) for subdf in annotations.groupby('class')])

    # print(annotations['class'].value_counts())








    train = pd.concat([subdf.sample(int(len(subdf)*TRAIN_SPLIT), random_state=42) for _, subdf in annotations.groupby('filename')])
    val = annotations.drop(train.index)

    print('Train:')
    print(train['class'].value_counts())

    print('Test:')
    print(val['class'].value_counts())

    for filename, samples in train.groupby('filename'):
        
        base_filename = os.path.basename(filename)
        shutil.copy(
            filename,
            os.path.join(
                'processed_dataset',
                'train',
                'images',
                base_filename
            )
        )

        filename_wo_ext = os.path.splitext(base_filename)[0]
        for _, row in samples.iterrows():
            write_annotation(
                os.path.join(
                    'processed_dataset',
                    'train',
                    'labels',
                    f'{filename_wo_ext}.txt'
                ),
                row
            )

    for filename, samples in val.groupby('filename'):
        
        base_filename = os.path.basename(filename)
        shutil.copy(
            filename,
            os.path.join(
                'processed_dataset',
                'val',
                'images',
                base_filename
            )
        )

        filename_wo_ext = os.path.splitext(base_filename)[0]
        for _, row in samples.iterrows():
            write_annotation(
                os.path.join(
                    'processed_dataset',
                    'val',
                    'labels',
                    f'{filename_wo_ext}.txt'
                ),
                row
            )





