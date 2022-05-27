from werkzeug.security import generate_password_hash

class Generator_password():
    FILE = 'local_aplication/config/general/config_password.json'

    import random   
    import json

    def __read_config(self) -> None:
        data = dict()
        with open(self.FILE) as f:
            data = self.json.load(f)
            #print(data)
        
        self.minus = data['caracteres']
        self.mayus = self.minus.upper()
        self.numeros = data['numeros']
        self.simbolos = data['simbolos']

        self.cantidad_mayusculas = data['cantidad_mayusculas']
        self.cantidad_minusculas = data['cantidad_minusculas']
        self.cantidad_numeros = data['cantidad_numeros']
        self.cantidad_simbolos = data['cantidad_simbolos']        

    def get_password(self) -> None:
        self.__read_config()

        mayus = self.random.sample(self.mayus, self.cantidad_mayusculas)
        minus = self.random.sample(self.minus, self.cantidad_minusculas)
        numeros = self.random.sample(self.numeros, self.cantidad_numeros)
        simbolos = self.random.sample(self.simbolos, self.cantidad_simbolos)

        self.longitud = self.cantidad_mayusculas \
            + self.cantidad_minusculas \
                + self.cantidad_numeros \
                    + self.cantidad_simbolos

        self.base = minus + mayus + numeros + simbolos

        muestra = self.random.sample(self.base, self.longitud) # desordenar los caracteres respetando el tamaño del string
        password = "".join(muestra)
        password_encriptado = generate_password_hash(password)
        password_encriptado = password_encriptado.split(":")
        self.resume = {
                    "basic" : password, 
                    password_encriptado[1]: password_encriptado[2]
                }
        return self.resume


if __name__ == "__main__":
    gen = Generator_password()
    print(gen.get_password())

  
    