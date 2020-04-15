from django.db.models import Q
from django.views.generic import TemplateView, ListView

from .models import City

from codesearch.src.predict import *
from codesearch.src.train import *

print("sys path values \n", sys.path)
#sys.path.insert(0, "E:\\Projects\\Django\\my_dj_apps\\my_code_search_djapp\\codesearch\\src\\utils")
sys.path.insert(0, "E:\\Projects\\Django\\my_dj_apps\\my_code_search_djapp\\codesearch\\src")
print("sys path values \n", sys.path)

class HomePageView(TemplateView):
    template_name = 'home.html'
    #def getHome(self) :
    #    return template_name

class SearchResultsView(ListView):
    model = City
    template_name = 'search_results.html'
    
    def get_queryset(self): 
        query = self.request.GET.get('q')
        #run()
        print("in get queryset query ", query)
        predictions = get_similar_code(query)
        print("predictions ", predictions)
        object_list = City.objects.filter(Q(name__icontains=query) | Q(state__icontains=query))
        print("object list ", object_list)
        return object_list