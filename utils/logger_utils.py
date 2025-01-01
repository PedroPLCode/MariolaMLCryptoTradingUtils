import json
from datetime import datetime

log_filename = None

def initialize_logger(settings_filename):
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
    if log_filename is None:
        raise RuntimeError("Logger has not been initialized. Call initialize_logger() first.")
    
    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{formatted_now}] {message}"
    
    print(log_message)
    
    with open(log_filename, 'a') as log_file:
        log_file.write(log_message + '\n')