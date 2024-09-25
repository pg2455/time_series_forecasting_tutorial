import subprocess
import os
from IPython.core.magic import register_cell_magic
from IPython.display import display, Markdown

@register_cell_magic
def load_file(line, cell):
    """
    Load the content of a file and insert it into a new cell below the current one.
    If the file is zipped, it will attempt to unzip it with the provided password.
    
    Usage in Jupyter:
    %%load_file [password]
    path/to/your/file.txt
    """
    lines = cell.strip().split('\n')
    if len(lines) != 1:
        print("Usage: Provide only the filename on a single line.")
        return
    
    filename = lines[0]
    password = line.strip() if line else None
    
    # Check if there's a zip file with the same name
    zip_filename = f"{filename}.zip"
    if not os.path.exists(filename) and os.path.exists(zip_filename):
        if not password:
            print("Password required to unzip the file.")
            return
        try:
            # Use unzip command with password
            cmd = ['unzip', '-P', password, zip_filename]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(result.stderr)
            print(f"File '{filename}' has been unzipped successfully.")
        except Exception as e:
            print(f"An error occurred while unzipping the file: {e}")
            return
    
    # Read the content of the file
    try:
        with open(filename, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return
    
    # Get the IPython shell
    ip = get_ipython()
    
    # Create a new cell with the file content
    ip.set_next_input(content)
    
    print(f"Content of '{filename}' has been loaded into a new cell below.")
    
    # Display a message in Markdown for better visibility
    display(Markdown(f"**File loaded:** `{filename}`"))

def zip_encrypt_file(filename, password):
    """
    Zip and encrypt a file with the given password using the zip command.
    
    :param filename: Name of the file to zip and encrypt
    :param password: Password for encryption
    """
    zip_filename = f"{filename}.zip"
    try:
        # Use zip command with password
        cmd = ['zip', '-P', password, zip_filename, filename]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        print(f"File '{filename}' has been zipped and encrypted as '{zip_filename}'.")
    except Exception as e:
        print(f"An error occurred while zipping and encrypting the file: {e}")

def register_magic():
    """
    Function to register the magic command.
    """
    get_ipython().register_magic_function(load_file, 'cell')