from roboflow import Roboflow
import os

os.environ['ROBOFLOW_API'] = 'MyrR6UFnOMOjR8gHw6MI'
rf = Roboflow(os.environ['ROBOFLOW_API'])
project = rf.workspace().project('waterdrain')
print('Project classes:', project.classes)
info = project.info()
print('Project info:', info)