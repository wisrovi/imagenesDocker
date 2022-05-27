# ********************************   FORMATO DATOS   ************************************
class Users(object):
    web = "web"
    admin = "admin"
    api = "api_rest"

class Password(object):
    basic = "basic"
    sha256 = "sha256"

    def __init__(self, bas=None, sha=None) -> None:
        if bas is not None:
            self.basic = bas
        if sha is not None:
            self.sha256 = sha

    def do_nothing(self):
        pass 

class Columns(object):
    date_create = "date_create"
    origin = "origin"
    author = "author"
    password = "password"
    
    basic = "basic"
    sha256 = "sha256"

class Data(object):
    date_create = None
    origin = None
    author = None
    password = str()

    import datetime
    def get_date(self):
        return self.datetime.datetime.utcnow()
    
    def __init__(self, author:str, password:str, origin=Users.api) -> None:
        self.origin = origin
        self.author = author
        self.password = password
        self.date_create = self.get_date()

    def do_nothing(self):
        pass 