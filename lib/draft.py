import sys
sys.path.append("..")


from lib.utils import train_parse_args


args = train_parse_args()
print(123)
d = vars(args)
print(d, type(d))

