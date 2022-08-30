from django.contrib.auth.models import User
from django.db import models


# Create your models here.

class Serial(models.Model):
    mac = models.CharField(max_length=20)
    serial = models.CharField(max_length=50)
    project = models.CharField(max_length=20)
    path_instalation = models.CharField(max_length=250)
    created = models.DateTimeField(auto_now_add=True)
    end = models.DateTimeField(auto_now=True)
    #user = models.ForeignKey(User, on_delete=models.CASCADE)
