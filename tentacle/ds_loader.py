import linecache
import numpy as np

class DatasetLoader(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self._cursor = 0
        self._wane = False

    def load(self, amount):
        content = []

        from_begin = self._cursor == 0
        rewind = False
        s = 0
        while s < amount:
            line = linecache.getline(self.file_name, self._cursor + 1)
            if line:
                line = line.rstrip().split(',')
                content.append([float(i) for i in line])
                self._cursor += 1
                s += 1
            else:
                if from_begin:
                    self._wane = True
                self._cursor = 0
                rewind = True
                if self._wane:
                    break
                
        self._has_more = False
        if not rewind:
            line = linecache.getline(self.file_name, self._cursor + 1)
            if line:
                self._has_more = True

        content = np.array(content)
        np.random.shuffle(content)
        return content, self._has_more
    
    @property
    def is_wane(self):
        return self._wane
