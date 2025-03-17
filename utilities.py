import os

def directory_making(path):
        access = 0o755
        try:
            if not os.path.exists(path):
                os.makedirs(path, access)
                return path
        except OSError:
            return None
        else:
            return path   