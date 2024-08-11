"""
WSGI config for backend project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
# Set the default settings module for the 'DJANGO_SETTINGS_MODULE' environment variable.
# This tells Django which settings file to use for the application.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# Get the WSGI application callable for the project.
# This application object can be used by a WSGI server to forward requests to your Django application.
application = get_wsgi_application()
