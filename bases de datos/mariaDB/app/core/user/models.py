from django.contrib.auth.models import AbstractUser
from django.db import models


# Create your models here.

"""
Class for replace the orginal class user of django
"""


class User(AbstractUser):
    token = models.CharField(max_length=250, default="")

    def get_token(self):
        if self.token:
            return self.token
        else:
            return ""



