import linecache
import os
import numpy as np


class DatasetLoader:

    def __init__(self, file_name):
        self.file_name = file_name
        self._cursor = 0
        self._wane = False
        self._file_size = self._get_file_size()

    def _get_file_size(self):
        return os.path.getsize(self.file_name) if os.path.exists(self.file_name) else 0

    def load(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive, got %r" % (amount,))

        linecache.checkcache(self.file_name)
        content = []

        from_begin = self._cursor == 0
        if from_begin:
            self._wane = False
        rewind = False
        s = 0
        while s < amount:
            line = linecache.getline(self.file_name, self._cursor + 1)
            if line:
                stripped = line.strip()
                if not stripped:
                    self._cursor += 1
                    continue

                fields = stripped.split(",")
                content.append([float(i) for i in fields])
                self._cursor += 1
                s += 1
            else:
                if from_begin:
                    self._wane = True
                self._cursor = 0
                rewind = True
                if from_begin:
                    break

        self._has_more = False
        if not rewind:
            line = linecache.getline(self.file_name, self._cursor + 1)
            if line:
                self._has_more = True

        self._file_size = self._get_file_size()

        content = np.array(content)
        np.random.shuffle(content)
        return content, self._has_more

    @property
    def is_wane(self):
        if self._wane and self._get_file_size() > self._file_size:
            self._wane = False
        return self._wane
