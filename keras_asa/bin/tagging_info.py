import sys

_, file = sys.argv

with open(file, "r") as f:
	data = f.readlines()

header = data[0].rstrip()

header = header + ",speaker,sentence_type,isStory"

print(header)

for i in range(1, len(data)):
	line = data[i].rstrip()

	items = line.split(",")

	spk, sent_type = items[0].split("_")

	if ".wav" in sent_type:
		sent_type = sent_type.replace(".wav", "")

	story = ["The-North-Wind", "They-Agreed-That", "Then-The-North", "Then-The-Sun", "And-So-The"]

	if sent_type in story:
		isStory = "story"
	else:
		isStory = "sentence"

	line = line + "," + spk + "," + sent_type + "," + isStory

	print(line)