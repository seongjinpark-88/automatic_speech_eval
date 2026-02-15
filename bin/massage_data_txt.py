with open("../results/mfcc_10CV_fluency.txt", "r") as f:
	data = f.readlines()


KOR = ["S19", "S23", "S24", "S25", "S28"]
ENG = ["S03", "S05", "S07", "S08", "S09"]
CHN = ["S02", "S04", "S21", "S22", "S26"]


for i in range(len(data)):
	(CV, stimuli, true, pred) = data[i].rstrip().split("\t")

	(subject, filename) = stimuli.split("_")

	if subject in KOR:
		language = "KOR"
	elif subject in ENG:
		language = "ENG"
	else:
		language = "CHN"

	result = "%s\t%s\t%s\t%s\t%s" % (CV, stimuli, language, true, pred)

	print(result)
