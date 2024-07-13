import csv

def parse_labels_file(file_path):
    id_to_sequence = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=' ')
        next(reader)  # Skip header
        for row in reader:
            class_id = int(row[0])
            cangjie_sequence = row[4]  # Assuming the Cangjie sequence is in the 4th column
            id_to_sequence[class_id] = cangjie_sequence
    return id_to_sequence

id_to_sequence = parse_labels_file(r'D:\cinnamon\week3\pytorch-cifar100\raw\labels.txt')
