import csv


def save_csv(data, filename='untitled.csv'):
    file = open(filename, 'w')
    with file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f'saved {filename}')

# EXAMPLE:
""" save_csv([["first_name", "second_name", "Grade1"],
           ['Alex', 'Brian', 'A'],
           ['Tom', 'Smith', 'B']], './data/circles/test.csv')  """
