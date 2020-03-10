#!/usr/bin/python

def sum_list(my_list): 
	sum_list = 0
	for i in my_list:
		sum_list = sum_list + i
	return(sum_list)

def mean_list(my_list):
	num_item = len(my_list)
	sum_item = sum_list(my_list)
	mean_value = sum_item / num_item
	return mean_value

def stdv_list(my_list):
	num_value = len(my_list)
	average_value = mean_list(my_list)
	sum_diff = 0
	for i in my_list:
		diff = (i - average_value) ** 2
		sum_diff = sum_diff + diff
	stdv = (sum_diff / num_value) ** 0.5
	return stdv

def nPVI(my_list):
	num_value = len(my_list)
	sum_value = 0
	for i in range(0,num_value-1):
		diff = my_list[0] - my_list[1]
		mean = (my_list[0] + my_list[1]) / 2
		result = abs(diff / mean)
		sum_value = sum_value + result

	nPVI = (100 * sum_value) / (num_value - 1)
	return(nPVI)

def rPVI(my_list):
	num_value = len(my_list)
	sum_value = 0
	for i in range(0,num_value-1):
		diff = abs(my_list[0] - my_list[1])
		sum_value = sum_value + diff

	rPVI = sum_value / (num_value - 1)
	return(rPVI)

import os
path = "../align_data"
files = os.listdir(path)

my_dict = {}

out = open("../results/rhythm.csv", "w")
out.write("fileName,%V,deltaV,deltaC,VarcoV,VarcoC,nPVI-V,rPVI-C\n")

for file in files:
	if (file[-4:] == ".PHN"):
		fileName = path + "/" + file
		my_lab = open(fileName, "r")
		data = my_lab.readlines()

		vowlList = []
		consList = []

		total_dur = 0

		vowel = ["a", "e", "i", "o", "u"]

		for line in data:
			line = line.replace("\n","")
			line = line.lower()
			items = line.split()

			phone = items[2]

			duration = (int(items[1]) - int(items[0])) / (16000)

			total_dur += duration

			if (any(i in phone for i in vowel) and phone != "sil"):
				vowlList.append(duration)
			elif(phone != "sil"):
				consList.append(duration)

		perc_v = (sum_list(vowlList) / total_dur) * 100
		delta_v = stdv_list(vowlList)
		delta_c = stdv_list(consList)
		varco_v = 100 * delta_v / mean_list(vowlList)
		varco_c = 100 * delta_c / mean_list(consList)
		nPVI_v = nPVI(vowlList)
		rPVI_c = rPVI(consList)

		out.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (file,perc_v,delta_v,delta_c,varco_v,varco_c,nPVI_v,rPVI_c))

out.close()