import base64
import json
import os
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import AES


class AES_Union:
    BLOCK_SIZE = 32  # Bytes
    key = str()

    def __init__(self, key: str, msg: str):
        self.key = key.encode('ascii')
        self.msg = msg.encode('ascii')

        self.cipher = AES.new(self.key, AES.MODE_ECB)

    def encript(self):
        self.msg_encript = self.cipher.encrypt(pad(self.msg, self.BLOCK_SIZE))
        self.msg_encript = base64.b64encode(self.msg_encript)
        self.msg_encript = self.msg_encript.decode('ascii')
        return self.msg_encript

    def decrypt(self, data: str = None):
        if data is None:
            data = self.msg_encript
        data = data.encode('ascii')
        data = base64.decodebytes(data)
        msg_dec = self.cipher.decrypt(data)
        data_bytes = unpad(msg_dec, self.BLOCK_SIZE)

        return data_bytes.decode('ascii')
