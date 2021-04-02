from django.contrib import admin

# Register your models here.
from .models import Question


class ChoiceInline(admin,StackedInline):
    model = Choice
    extra = 3

class QuestionAdmin(admin.ModelAdmin):
    fieldsets = [
        (None, {"fields": ['question_text']}),
        ('Date information', {'fields':['pub_date'], "classes" : ['collapse']})
    ]

admin.site.register(Question, QuestionAdmin)


## Choice 페이지 관련 admin 게정만들기