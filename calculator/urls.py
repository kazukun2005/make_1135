from django.urls import path
from . import views

urlpatterns = [
    # メインページを表示するビュー
    path('', views.index_view, name='index'),
    # 計算処理を行うビュー (HTMXがアクセスする)
    path('calculate/', views.calculate_view, name='calculate'),
]