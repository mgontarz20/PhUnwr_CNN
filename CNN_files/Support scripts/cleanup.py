import os
import shutil
path = r"C:\Users\Michał\Desktop\DO RAPORTU\21.10\old"

folders = next(os.walk(path))[1]
print(folders)
for folder in folders:
    files = next(os.walk(os.path.join(path, folder)))[2]
    if not files:
        shutil.rmtree(os.path.join(path,folder))