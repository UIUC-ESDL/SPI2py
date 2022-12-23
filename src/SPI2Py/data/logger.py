"""


"""
import numpy as np

def extract_design_vectors():
    pass


with open('examples/demo_1/design_vector.log') as f:
    outputs = f.read()

outputs = outputs.replace('\n', ' ')

# outputs = outputs.split('[]')

# outputs = '[' + outputs + ']'

# outputs = np.fromstring(outputs)

# wrong shape...

outputs = outputs.replace('] [', '] \n [')



outputs = outputs.replace('[', '')
outputs = outputs.replace(']', '')

outputs = outputs.split('\n')

design_vectors = []

for row in outputs:
    # Get rid of excessive whitespaces
    row = row.replace('  ', ' ')
    design_vector = [i for i in row.split(' ')]
    design_vectors.append(design_vector)
