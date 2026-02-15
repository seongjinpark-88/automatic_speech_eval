with open("../results/mel_50_fluency_regression.txt", "r") as f:
	data = f.readlines()


KOR = ["S19", "S23", "S24", "S25", "S28"]
ENG = ["S03", "S05", "S07", "S08", "S09"]
CHN = ["S02", "S04", "S21", "S22", "S26"]


for line in data:
	(file, true, prediction) = line.rstrip().split("\t")

	(subject, filename) = file.split("_")

	if subject in KOR:
		language = "KOR"
	elif subject in ENG:
		language = "ENG"
	else:
		language = "CHN"

	result = "%s\t%s\t%s\t%s" % (file, language, true, prediction)

	print(result)
