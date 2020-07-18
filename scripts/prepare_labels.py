from argparse import ArgumentParser
from .constants import PATHS


def parse_args(paths_default):
    parser = ArgumentParser()
    parser.add_argument(
        '--new_indices_labels_filename',
        default=paths_default['new_indices_labels_filename'],
        type=str,
        help=f"new indices labels filename, default: {paths['new_indices_labels_filename']}")
    parser.add_argument(
        '--new_original_indices_bijection_filename',
        default=paths_default['new_original_indices_bijection_filename'],
        type=str,
        help=f"new-original indices bijection filename, default: {paths['new_original_indices_bijection_filename']}")
    parser.add_argument(
        '--original_indices_labels_filename',
        default=paths_default['original_indices_labels_filename'],
        type=str,
        help=f"original indices labels filename, default: {paths['original_indices_labels_filename']}")

    return parser.parse_args()


def prepare_labels(paths):
    new_indices_labels_file = open(paths['new_indices_labels_filename'], 'r')
    new_original_indices_bijection_file = open(paths['new_original_indices_bijection_filename'], 'r')
    original_indices_labels_file = open(paths['original_indices_labels_filename'], 'w')

    new_indices_labels_file.readline()
    new_indices_labels_file.readline()
    new_original_indices_bijection_file.readline()

    original_indices_labels_file.write("orig_filename\tlabels")

    for i in range():#TODO
        labels_string = new_indices_labels_file.readline()
        bijection_string = new_original_indices_bijection_file.readline()

        labels = labels_string.split()[1:]
        orig_filename = bijection_string.split()[2] #TODO think about indices

        original_indices_labels_file.write(orig_filename, end=' ')
        original_indices_labels_file.write(' '.join(labels), end=' ')


def main():
    paths_default = PATHS
    paths = vars(parse_args(paths_default))
    prepare_labels(paths)


if __name__ == "__main__":
    main()

