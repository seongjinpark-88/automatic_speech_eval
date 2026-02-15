from collections import defaultdict

with open("../data/accent_1.csv") as f:
	data_1 = f.readlines()

with open("../data/accent_2.csv") as f:
	data_2 = f.readlines()

with open("../data/validation_accented_stim.csv") as f:
	val = f.readlines()


val_list = {}
for i in range(1, len(val)):
	line = val[i].rstrip().split(",")

	stim = line[0]
	val_list[stim] = 1


total_num_stim = len(list(val_list.keys()))
data_1_num = 51
data_2_num = total_num_stim - data_1_num

subject_val_data1 = defaultdict(float)
subject_val_data2 = defaultdict(float)
data_stim = []

for i in range(1, len(data_1)):

	line = data_1[i].rstrip().split(",")
	if (len(line) > 3):
		sub_id = line[3]
		score = line[-6]
		stimuli = line[-4]

		if stimuli in val_list:
			data_stim.append(stimuli)
			if int(score) >= 4:
				subject_val_data1[sub_id] += 1

for i in range(1, len(data_2)):

	line = data_2[i].rstrip().split(",")
	if (len(line) > 3):
		sub_id = line[3]
		score = line[-6]
		stimuli = line[-4]

		if stimuli in val_list:
			data_stim.append(stimuli)
			if int(score) >= 4:
				subject_val_data2[sub_id] += 1

print(total_num_stim)


from pprint import pprint

pprint(subject_val_data1)
pprint(subject_val_data2)

from collections import Counter
pprint(Counter(data_stim))
pprint(len(list(Counter(data_stim).keys())))