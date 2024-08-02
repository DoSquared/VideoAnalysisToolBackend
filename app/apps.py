#This file defines the configuration settings for a Django application named 'app'
#specifying that the default primary key field type for models should be BigAutoField

from django.apps import AppConfig

# Define a configuration class for the Django application
class AppConfig(AppConfig):
    # Set the default primary key field type for models to BigAutoField
    default_auto_field = 'django.db.models.BigAutoField'
    # Specify the name of the application
    name = 'app'
