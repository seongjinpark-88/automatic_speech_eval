with open("../accented_data.csv", "r") as f:
	data = f.readlines()


KOR = ["S19", "S23", "S24", "S25", "S28"]
ENG = ["S03", "S05", "S07", "S08", "S09"]
CHN = ["S02", "S04", "S21", "S22", "S26"]


for i in range(len(data)):
	(true, title, filename) = data[i].rstrip().split(",")

	(subject, filename) = title.split("_")

	if subject in KOR:
		language = "KOR"
	elif subject in ENG:
		language = "ENG"
	else:
		language = "CHN"

	result = "%s\t%s\t%s\t%s" % (true, language, title, filename)

	print(result)
