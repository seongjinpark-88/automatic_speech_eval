import sys

with open(sys.argv[1], "r") as f:
	data = f.readlines()

for i in range(0, 46):
	sent = data[i-1].rstrip()
	sent = sent.replace("'", "")
	sent = sent.replace("\"", "")
	sent = sent.replace(",", "")
	words = sent.lower().split()

	new_name = "%s-%s-%s" % (words[0].capitalize(), words[1].capitalize(), words[2].capitalize())

	print(new_name)