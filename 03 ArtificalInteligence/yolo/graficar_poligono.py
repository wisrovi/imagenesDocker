import cv2
import numpy as np
from shapely.geometry import Polygon

# Definir las coordenadas del polígono en forma de casa
house_vertices = [(30, 70), (70, 70), (70, 30), (50, 10), (30, 30)]

# Crear un polígono de Shapely que representa la casa
house = Polygon(house_vertices)

# Obtener los límites del polígono
min_x, min_y, max_x, max_y = house.bounds

# Crear una imagen en blanco que se ajuste a los límites del polígono
image = np.zeros((int(max_y - min_y), int(max_x - min_x), 3), dtype=np.uint8)

# Crear una cuadrícula en el polígono
step = 5  # Espaciado entre las líneas de la rejilla

for x in range(int(min_x), int(max_x), step):
    cv2.line(image, (x - int(min_x), 0), (x - int(min_x), int(max_y - min_y)), (0, 0, 255), 1)

for y in range(int(min_y), int(max_y), step):
    cv2.line(image, (0, y - int(min_y)), (int(max_x - min_x), y - int(min_y)), (0, 0, 255), 1)

# Dibujar el polígono (casa)
polygon_points = [(int(x - min_x), int(y - min_y)) for x, y in house.exterior.coords]
points = np.array(polygon_points, dtype=np.int32)
cv2.fillPoly(image, [points], (0, 255, 0))

cv2.imshow('Casa con Grid Fill', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
