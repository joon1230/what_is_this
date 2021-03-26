from django.db import models

# Create your models here.

from django.db import models
# models : https://docs.djangoproject.com/ko/3.1/ref/models/instances/#django.db.models.Model


# table, database의 컬럼(속성) 들을 정의하는 구간
class Question(models.Model):
    question_text = models.CharField(max_length=200) #
    pub_date = models.DateTimeField("date published")


class Choice(models.Model):
    question = models.ForeignKey( Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)