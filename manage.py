#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os  # Import the os module to interact with the operating system.
import sys  # Import the sys module to access system-specific parameters and functions.

def main():
    """Run administrative tasks."""
    # Set the default settings module for Django. This specifies which settings file Django should use.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    try:
        # Import Django's command-line execution function.
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        # If Django cannot be imported, raise an error with a helpful message.
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    # Execute the command-line utility with the arguments provided from the command line.
    execute_from_command_line(sys.argv)

# This ensures that the main() function is called only when this script is executed directly.
if __name__ == '__main__':
    main()
