import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

def get_parsed_arguments(first_arg_string, second_arg_string=None):
    """
    Parses and returns one or two command-line arguments.

    This function uses the argparse library to parse one required positional argument 
    and one optional positional argument provided via the command line. Each argument 
    is accompanied by a descriptive message passed to the function.

    Parameters:
        first_arg_string (str): A description for the first argument.
        second_arg_string (str, optional): A description for the second argument. Defaults to None.

    Returns:
        tuple: A tuple containing the parsed values of the arguments. The second argument is None if not provided.
    """
    try:
        
        parser = argparse.ArgumentParser(description="A script that accepts one or two arguments.")
        parser.add_argument(
            'first_argument',
            type=str,
            help=f"A required argument. {first_arg_string}"
        )
        parser.add_argument(
            'second_argument',
            type=str,
            nargs='?',
            default=None,
            help=f"An optional argument. {second_arg_string or 'No description provided.'}"
        )

        args = parser.parse_args()
        first_argument = args.first_argument
        second_argument = args.second_argument

        return first_argument, second_argument
    
    except Exception as e:
        print(e)
        return None