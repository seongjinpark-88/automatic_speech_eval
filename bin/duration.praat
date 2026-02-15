# Original script is from praat scripts in Internet (4.7.2003 Mietta Lennes)
# Modified by Seongjin 12.21.2014

form Analyze formant values, durations from labeled segments in files
	comment Directory of sound files
	text sound_directory ./extract/S19/wav/
	sentence Sound_file_extension .wav
	comment Directory of TextGrid files
	text textGrid_directory ./extract/S19/wav/
	sentence TextGrid_file_extension .TextGrid
	comment Full path of the resulting text file:
	# save results as a csv file.
	text resultfile ./extract/
	comment Which tier do you want to analyze?
	sentence Tier target/word
endform

# Here, you make a listing of all the sound files in a directory.
# The example gets file names ending with ".wav" from D:\tmp\

Create Strings as file list... list 'sound_directory$'*'sound_file_extension$'
numberOfFiles = Get number of strings

# Check if the result file exists:
if fileReadable (resultfile$)
	pause The result file 'resultfile$' already exists! Do you want to overwrite it?
	filedelete 'resultfile$'
endif

# Write a row with column titles to the result file:

titleline$ = "file,time_onset,time_offset,word'newline$'"
fileappend "'resultfile$'" 'titleline$'

# Go through all the sound files, one by one:

for ifile to numberOfFiles
	select Strings list
	filename$ = Get string... ifile
	# A sound file is opened from the listing:
	Read from file... 'sound_directory$''filename$'
	soundname$ = selected$ ("Sound", 1)
	# Open a TextGrid with the same name:
	gridfile$ = "'textGrid_directory$''soundname$''textGrid_file_extension$'"
	if fileReadable (gridfile$)
		Read from file... 'gridfile$'
		# Find the tier number that has the label given in the form:
		call GetTier 'tier$' tier
		numberOfIntervals = Get number of intervals... tier
		# Pass through all intervals in the selected tier:
		for interval to numberOfIntervals
			label$ = Get label of interval... tier interval
			if label$ <> ""
				start = Get starting point... tier interval
				end = Get end point... tier interval
				midpoint = (start + end) / 2
				# get the duration of that interval
				duration = (end - start)
				# Save result to text file:
				resultline$ = "'soundname$','start','end','label$''newline$'"
				fileappend "'resultfile$'" 'resultline$'
				select TextGrid 'soundname$'
			endif
		endfor
		# Remove the TextGrid object from the object list
		select TextGrid 'soundname$'
		Remove
	endif
	# Remove the temporary objects from the object list
	select Sound 'soundname$'
	Remove
	select Strings list
	# and go on with the next sound file!
endfor

Remove

# This procedure finds the number of a tier that has a given label.
# 'procedure' is another name for 'sub' in perl, or 'def' in python.

procedure GetTier name$ variable$
        numberOfTiers = Get number of tiers
        itier = 1
        repeat
                tier$ = Get tier name... itier
                itier = itier + 1
        until tier$ = name$ or itier > numberOfTiers
        if tier$ <> name$
                'variable$' = 0
        else
                'variable$' = itier - 1
        endif

	if 'variable$' = 0
		exit The tier called 'name$' is missing from the file 'soundname$'!
	endif

endproc