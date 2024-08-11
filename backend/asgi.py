"""
ASGI config for backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os  # Import the os module, which provides a way to interact with the operating system.

from django.core.asgi import get_asgi_application  # Import the ASGI application function from Django.

# Set the default settings module for the 'DJANGO_SETTINGS_MODULE' environment variable.
# This tells Django which settings file to use for the application.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# Get the ASGI application callable for the project.
# This application object can be used by an ASGI server to communicate with your Django application.
application = get_asgi_application()
