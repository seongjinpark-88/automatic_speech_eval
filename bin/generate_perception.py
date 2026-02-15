import sys
from collections import defaultdict
import random


def append_dict(data, dictionary, key, start, end):
	for i in range(start, end):
		sent = data[i].rstrip()
		dictionary[key].append(sent)

def create_list(spk, dictionary, final_dict):

	idx = 0

	for i in range(0, len(spk)):
		for j in range(0, 10):

			new_j = j + idx

			if new_j < 10:

				sents = dictionary[new_j]
			else:
				new_j = new_j - 10
				sents = dictionary[new_j]

			wavs = []

			for sent in sents:
				wav = '"' + spk[i] + "_" + sent + '",\n'
				wavs.append(wav)

			final_dict[j].extend(wavs)
		idx += 1



with open(sys.argv[1], "r") as f:
	data = f.readlines()


SPK = ["S03", "S09", "S05", "S07", "S08", "S02", "S21", "S22", "S04", "S26"]
SPK_ALL = ["S03", "S09", "S05", "S07", "S08", "S02", "S21", "S22", "S04", "S26", 
		   "S19", "S23", "S25", "S24", "S28"]

cell = defaultdict(list)

append_dict(data, cell, 0, 0, 4)
append_dict(data, cell, 1, 4, 8)
append_dict(data, cell, 2, 8, 12)
append_dict(data, cell, 3, 12, 16)
append_dict(data, cell, 4, 16, 20)
append_dict(data, cell, 5, 20, 24)
append_dict(data, cell, 6, 24, 28)
append_dict(data, cell, 7, 28, 32)
append_dict(data, cell, 8, 32, 36)
append_dict(data, cell, 9, 36, 40)


counter_list = defaultdict(list)

create_list(SPK, cell, counter_list)

# from pprint import pprint
# pprint(counter_list)

# for i in range(0, 10):
# 	print(len(counter_list[i]))


# for i in range(1, 11):
# 	print('\t"Transcription_questions_%s": {' % i)
# 	print('\t\t"type": "basic", ')
# 	print('\t\t"stimuli": [')
# 	wav_files = counter_list[i-1]
# 	file_list = "\t\t\t".join(wav_files)
# 	print("\t\t\t%s" % file_list)
# 	print('\t\t],')
# 	print('\t\t"responses": [')
# 	print('\t\t\t"Transcription"')
# 	print("\t\t]")
# 	print("\t},")

for s in SPK_ALL:
	for sent in data:
		sent = sent.rstrip()
		wav = '' + s + "_" + sent + ''
		result = wav + ",audio," + wav + ".mp3,false"
		print(result)

# all_files = []
# for spk in SPK_ALL:
# 	for sent in data:
# 		sent = sent.rstrip()
# 		result = '"' + spk + '_' + sent + '"'
# 		all_files.append(result)


# shuffle_list = []
# shuffle_list.extend(all_files)
# random.shuffle(shuffle_list)
# print(len(shuffle_list))
# # print(all_files)
# # print(shuffle_list)

# indices = [0, 100, 200, 300, 400, 500]

# for i in range(0, len(indices)):
# 	indice = i + 1
# 	print('\t"Scale_questions_%d": {' % indice)
# 	print('\t\t"type": "basic",')
# 	print('\t\t"stimuli": [')
# 	items = ",\n\t\t\t".join(shuffle_list[indices[i]:indices[i] + 100])
# 	print('\t\t\t%s' % items)
# 	print('\t\t],')
# 	print('\t\t"stimulus_pattern": {"order": "random"},')
# 	print('\t\t"responses": ["Scale_response"],')
# 	print('\t\t"response_confirm": true')
# 	print('\t},')


