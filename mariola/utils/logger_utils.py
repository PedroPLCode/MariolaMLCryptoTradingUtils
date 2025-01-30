import json
from datetime import datetime

log_filename = None

def initialize_logger(settings_filename):
    """
    Initializes the logger by loading the log file path from a JSON settings file.

    Reads the `settings_filename` JSON file to extract the `log_filename` value, 
    which determines where log messages will be written. If the file does not exist, 
    is invalid JSON, or does not contain the required key, the program exits with an error message.

    This function modifies the global variable `log_filename` with the log file path.

    Parameters:
        settings_filename (str): The path to the JSON settings file.

    Raises:
        FileNotFoundError: If the `settings_filename` file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If the `log_filename` key is missing in the JSON.

    Returns:
        None
    """
    global log_filename
    
    try:
        with open(settings_filename, 'r') as f:
            settings_data = json.load(f)
            log_filename = settings_data['settings']['log_filename']
            
    except FileNotFoundError:
        print(f"Error: File {settings_filename} not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {settings_filename} is not a valid JSON.")
        exit(1)
    except KeyError:
        print("Error: log_filename not found in settings.json.")
        exit(1)


def log(message):
    """
    Logs a message to the console and appends it to the log file.

    This function writes the provided message to both the console and a log file, 
    along with the current timestamp. The logger must be initialized with 
    `initialize_logger()` before calling this function.

    Parameters:
        message (str): The log message to be recorded.

    Raises:
        RuntimeError: If the logger has not been initialized by calling `initialize_logger()` first.

    Returns:
        None
    """
    try:
        if log_filename is None:
            raise RuntimeError("Logger has not been initialized. Call initialize_logger() first.")
        
        now = datetime.now()
        formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{formatted_now}] {message}"
        
        print(log_message)
        
        with open(log_filename, 'a') as log_file:
            log_file.write(log_message + '\n')
        
    except Exception as e:
        print(e)
        return None