import sys
from collections import Counter

def add_count(counter_name, segments):
	if segments != "":
		if "," in segments:
			segs = segments.replace(" ", "")
			segs = segs.split(",")

			for seg in segs:
				counter_name[seg] += 1

		else:
			counter_name[segments] += 1


_, file = sys.argv

with open(file, "r") as f:
	data = f.readlines()


ch_count = Counter()
kr_count = Counter()
for i in range(1, len(data)):

	items = data[i].rstrip().split(",")
	group = items[-1]
	lengthen_item = items[-2]

	if group == "KOR":
		add_count(kr_count, lengthen_item)
	elif group == "CHN":
		add_count(ch_count, lengthen_item)
from pprint import pprint

pprint(ch_count)