# # from django.urls import path
# # from . import views
# # urlpatterns = [
# #     path('', views.index, name='index'),
# #     path('regLog_details/', views.regLog_details, name='regLog_details'),
# #     path('regLog_atelier/', views.regLog_atelier, name='regLog_atelier'),
# #     path('regLog_tester/', views.regLog_tester, name="regLog_tester"),
# #     path('reglog_prediction', views.regLog_prediction, name='reglog_prediction')
# # ]
# from django.urls import path
# from . import views

# urlpatterns = [
#     # Page d'accueil avec toutes les cartes
#     path('', views.dashboard, name='dashboard'),
    
#     # URL dynamique : capture le nom du modèle (ex: log_class, rf_reg)
#     path('tester/<str:algo_name>/', views.tester_modele, name='tester_modele'),
# ]
from django.urls import path
from . import views

urlpatterns = [
    # Page d'accueil
    path('', views.index, name='index'),
    
    # Pages de détails (Statiques)
    path('details/regression-logistique/', views.regLog_details, name='regLog_details'),
    
    # Page de Test DYNAMIQUE (La plus importante)
    # Exemple d'appel : /tester/log_class/ ou /tester/rf_reg/
    path('tester/<str:algo_name>/', views.tester_modele, name='tester_modele'),
    path('details/dt-classification/', views.dt_class_details, name='dt_class_details'),
    path('details/rf-classification/', views.rf_class_details, name='rf_class_details'),
    path('details/svm/', views.svm_class_details, name='svm_details'),
    path('details/linear-regression/', views.lin_reg_details, name='lin_reg_details'),
    path('details/rf-regression/', views.rf_reg_details, name='rf_reg_details'),
    path('details/dt-regression/', views.dt_reg_details, name='dt_reg_details'),
    path('details/svm-regression/', views.svm_reg_details, name='svm_reg_details'),
    path('details/XGboost-classification/', views.xgb_classi_details, name='XGboost_class_details'),
    path('details/XGboost-regression/', views.xgb_reg_details, name='XGboost_reg_details'),
]
