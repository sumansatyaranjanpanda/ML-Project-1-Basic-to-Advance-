from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements,
    ignoring editable install markers like "-e ."
    '''
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.strip()
            if line and not line.startswith('-e'):
                requirements.append(line)
    return requirements

    
setup(

    name='mlproject',
    version='0.0.1',
    author='Raja Panda',
    author_email='satyaranjanpandasuman@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )