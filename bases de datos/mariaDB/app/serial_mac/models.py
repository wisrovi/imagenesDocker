from django.contrib.auth.models import User
from django.db import models


# Create your models here.

class Serial(models.Model):
    mac = models.CharField(max_length=20)
    project = models.CharField(max_length=20)
    path_instalation = models.CharField(max_length=250)
    created = models.DateTimeField(auto_now_add=True)
    end = models.DateTimeField(auto_now_add=True)
    hash_checksum = models.CharField(max_length=200, default="")
    hash_hash = models.CharField(max_length=200, default="")
    hash_md5 = models.CharField(max_length=200, default="")


class FileSincronizacion(models.Model):
    server = models.CharField(max_length=30)
    project = models.CharField(max_length=50)
    vence = models.DateTimeField(auto_now_add=False)
    user = models.CharField(max_length=20)
    password = models.CharField(max_length=250)
    active = models.BooleanField(default=True)

