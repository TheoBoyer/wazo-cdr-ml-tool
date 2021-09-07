"""

    TMP Folder wrapper

"""

import os
import shutil
import sys

class TMPFolder:
    """
        TMP Folder wrapper
    """
    def __init__(self, tmp_folder="tmp"):
        self.name = tmp_folder
        if os.path.exists(self.name):
            self.name = tmp_folder + "2"

    def __enter__(self):
        os.mkdir(self.name)
        os.chdir(self.name)
        sys.path.append("../")
        return self.name

    def __exit__(self, exc_type, exc_value, tb):
        os.chdir("../")
        shutil.rmtree(self.name)