# import os module

import os

# open file

with open("list4.csv") as f:
	data = f.readlines()

# function to rename multiple files


# for filename in os.listdir("./extract/S19"):
for i in range(1, 46):
	sent = data[i-1].rstrip()
	sent = sent.replace("'", "")
	sent = sent.replace("\"", "")
	sent = sent.replace(",", "")
	words = sent.lower().split()

	new_name = "extract/S24/wav/S24_%s-%s-%s.wav" % (words[0].capitalize(), words[1].capitalize(), words[2].capitalize())

	source = "extract/S24/wav/S24_%s.wav" % str(i)

	try:
		os.rename(source, new_name)
	except:
		pass
