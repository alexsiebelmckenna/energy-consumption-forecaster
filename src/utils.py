from os import listdir

def listdir_remove(path, string='.DS_Store', sort=True):
    # List directories in path and remove string(s) if string(s) exist in directory list
    # Intended for OS-specific files (e.g. .DS_Store on Macs, which is the default)
    listed = listdir(path)
    if string in listed:
        listed.remove(string)
    if sort:
        listed = sorted(listed)
    return listed