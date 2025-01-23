import os
import re

def create_dir(path):
    """
    Create a directory if it does not exist
    """
    os.makedirs(path, exist_ok=True)

def clean_text(text):
    """
    Clean text by
        1. Convert text to lower case
        2. Removing text within parentheses
        3. Removing text within square brackets
        4. Removing single and double quotes
        5. Removing space before comma
        6. Splitting text after a period and colon
    """
    text = text.lower()  # Lowercase text
    text = re.sub(r'\(.*?\)', '', text)  # Removes text within parentheses (including the parentheses)
    text = re.sub(r'\[.*?\]', '', text)  # Removes text within square brackets (including the brackets)
    text = re.sub(r'[\'\"]', '', text)  # Removes both single and double quotes
    text = re.sub(r'\s+,', ',', text) # Removes space before comma
    text = re.sub(r'\.\s*', '\n', text) # Split text after a period
    text = re.sub(r'\:\s*', '\n', text) # Split text after a colon
    return text

def setup_joeynmt():
    """
    Setup JoeyNMT by cloning the repository and installing the required packages
    """
    commands = """
        #Create and activate a virtual environment to install the package into:
        python -m venv jnmt
        source jnmt/bin/activate

        #Then clone JoeyNMT from GitHub and switch to its root directory:
        git clone https://github.com/joeynmt/joeynmt.git
        cd joeynmt
        # Install JoeyNMT and it’s requirements:
        pip install .

    """
    os.system(commands)

def writefile(filename, data):
    """
    Write data to a file
    """
    f = open(filename, 'w')
    for text in data:
        f.write(text.strip() + '\n')
    f.close()

def readfile(filename):
    """
    Read data from a file
    """
    f = open(filename, 'r')
    data = f.readlines()
    f.close()
    return data #[line.strip() for line in data]