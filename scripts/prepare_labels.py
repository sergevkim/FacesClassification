from argparse import ArgumentParser
from constants import PATHS


def parse_args(paths_default):
    parser = ArgumentParser()
    parser.add_argument(
        '--new_indices_labels_filename',
        default=paths_default['new_indices_labels_filename'],
        type=str,
        help=f"new indices labels filename, default: {paths_default['new_indices_labels_filename']}")
    parser.add_argument(
        '--new_original_indices_bijection_filename',
        default=paths_default['new_original_indices_bijection_filename'],
        type=str,
        help=f"new-original indices bijection filename, default: {paths_default['new_original_indices_bijection_filename']}")
    parser.add_argument(
        '--original_indices_labels_filename',
        default=paths_default['original_indices_labels_filename'],
        type=str,
        help=f"original indices labels filename, default: {paths_default['original_indices_labels_filename']}")

    return parser.parse_args()


def remember_original_images_labels(original_indices_labels_file):
    n_filenames = int(original_indices_labels_file.readline())
    label_types = original_indices_labels_file.readline()

    labels = dict()

    for i in range(n_filenames):
        labels_string = original_indices_labels_file.readline().split()
        filename = labels_string[0]
        labels[filename] = labels_string[1:]

    return labels


def prepare_labels(paths):
    original_indices_labels_file = open(paths['original_indices_labels_filename'], 'r')
    new_original_indices_bijection_file = open(paths['new_original_indices_bijection_filename'], 'r')
    new_indices_labels_file = open(paths['new_indices_labels_filename'], 'w')

    labels = remember_original_images_labels(original_indices_labels_file)

    new_original_indices_bijection_file.readline() # additional info
    new_original_indices_bijection_file.readline() # zero object string that does not exist

    new_indices_labels_file.write("new_filename  orig_filename  labels\n")

    for i in range(1, 30000):
        bijection_string = new_original_indices_bijection_file.readline()

        orig_filename = bijection_string.split()[2]
        new_filename = f"{i}.png" #TODO check

        labels_string = '  '.join(labels[orig_filename])
        labels_string = labels_string.replace('-1', '0')

        new_indices_labels_file.write(f"{new_filename}     {orig_filename}     {labels_string}\n")


def main():
    paths = vars(parse_args(PATHS))
    prepare_labels(paths)


if __name__ == "__main__":
    main()

