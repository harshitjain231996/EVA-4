import psutil
import numpy as np
import time

# Dependencies on following packages, to be set at applictaion code
'''
if pkgutil.find_loader("memory_profiler") is None:
  !pip install memory_profiler

if pkgutil.find_loader("line_profiler") is None:
  !pip install line_profiler
  
%load_ext line_profiler
%load_ext memory_profiler
'''

'''
show: it display memory statistic at the momnet it is called. delta represent the difference in memory usage between two subsequent show calls.
'''
class MemoryUsage():
  def __init__(self):
      self.last_used = psutil.virtual_memory().used

  def show(self, tag=None):
      mem_p = psutil.virtual_memory()
      used = mem_p.used
      delta = used - self.last_used
      self.last_used = used
      unit = 1e6

      #profile = {"unit": "MB", "tag": tag, "delta": np.round(delta/unit,4), "percent_used": mem_p.percent, "used": np.round(used/unit,4)}
      profile = {"delta": np.round(delta/unit,4), "percent_used": mem_p.percent, "used": np.round(used/unit,4)}
      return profile

'''
capture time difference between start and end/end_show fxn calls 
'''
class TimeProfile():
  def __init__(self):
      self.startTime = time.time()
      self.tag = ""

  def start(self, tag=None):
      self.startTime = time.time()
      self.tag = tag

  def end(self):
      seconds = time.time() - self.startTime
      return seconds

  def end_show(self):
      seconds = time.time() - self.startTime
      print(f'[{self.tag}]: Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))