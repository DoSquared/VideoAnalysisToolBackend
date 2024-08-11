from django.urls import path  # Import the path function to define URL patterns
from app.views import home, get_video_data, leg_raise_task, updatePlotData, update_landmarks  # Import view functions from the app's views module

# Define the URL patterns for the application
urlpatterns = [
    # URL pattern for the home page, mapped to the 'home' view function
    path('', home),
    
    # URL pattern for video data analysis, mapped to the 'get_video_data' view function
    path('video/', get_video_data),
    
    # URL pattern for leg raise task analysis, mapped to the 'leg_raise_task' view function
    path('leg_raise/', leg_raise_task),
    
    # URL pattern for updating plot data, mapped to the 'updatePlotData' view function
    path('update_plot/', updatePlotData),
    
    # URL pattern for updating landmarks, mapped to the 'update_landmarks' view function
    path('update_landmarks/', update_landmarks),
    
    # URL pattern for toe tap task analysis (same as leg_raise_task), mapped to the 'leg_raise_task' view function
    path('toe_tap/', leg_raise_task)
]
