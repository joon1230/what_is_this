# view 의 종류
# 하나의 페이지라 생각하면 편하다.

# view 의 종류
# class view : CRUD 기능을 상속하여 사용하기 용이하다. generic view
# function view :  단순한 view의 형태

from django.http import HttpResponse
def index(request):
    html = "<html><body>Hello django</body></html>"
    return HttpResponse(html)