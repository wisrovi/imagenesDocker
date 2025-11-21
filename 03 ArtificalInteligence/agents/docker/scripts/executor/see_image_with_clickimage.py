import sys
import climage

cantidad_carecteres = 80

# Si no pasas argumentos, usa una imagen por defecto o error
if len(sys.argv) < 2:
    print("Uso: python see_ascii.py <imagen>")
    sys.exit(1)

output = climage.convert(sys.argv[1], is_unicode=True, width=cantidad_carecteres)
# width=80 ajusta el ancho a caracteres de consola estándar

print(output)
