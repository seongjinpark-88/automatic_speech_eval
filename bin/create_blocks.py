
for i in range(1, 11):
	print('\t\t"Block%d" : {' % i)
	print('\t\t\t"trial_templates": ["Transcription_questions_%d"],' % i)
	print('\t\t\t"pattern": {"order": "random"}')
	print('\t\t},')

print('\t"block_sequence" : [')
print('\t\t{')
for i in range(1, 11):
	print('\t\t"Group%d": ["Block%d", "Block11"]' % (i, i), end = "")
	if i < 10:
		print(",")
	else:
		print()

print('\t\t}')
print('\t]')
