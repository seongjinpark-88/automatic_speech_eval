import sys
import os 

with open(sys.argv[1], "r") as f:
	data = f.readlines()

for line in data:
	items = line.rstrip().split()

	utt_id = items[0]
	phn_stt = items[2]
	phn_end = float(phn_stt) + float(items[3])
	phn = items[4]
	file = utt_id + ".txt"
	filename = os.path.join(sys.argv[2], file)
	with open(filename, "a") as out: 
		result = str(phn_stt) + "\t" + str(phn_end) + "\t" + phn + "\n"
		out.write(result)