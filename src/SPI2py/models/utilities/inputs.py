import tomli

def read_input_file(input_file_path):
    with open(input_file_path, 'rb') as f:
        input_file = tomli.load(f)

    return input_file