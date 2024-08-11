from django.shortcuts import render  # Import the render function from Django's shortcuts module

# Define a view function named 'home' that takes an HTTP request object as a parameter
def home(request):
    # The render function combines a given template with a context dictionary and returns an HttpResponse object
    # In this case, it renders the 'index.html' template without any additional context
    return render(request, template_name='index.html')
