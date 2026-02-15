## Praat Script for openning and modifying files easily.
## Open Sounds and TextGrids in same folder (with same name), and after modify, 
## you can move on to next files without saving
## 2015. 1. 13. Written by Seongjin

form Open files and Modify
	comment Sound information
	sentence snd_dir ./extract/S19/wav/
	sentence snd_ext .wav
	comment TextGrid information
	sentence txt_dir ./extract/S19/wav/
	sentence txt_ext .TextGrid
endform

## Sound file list
Create Strings as file list... snd_list 'snd_dir$'*'snd_ext$'
num_snd = Get number of strings

## Main program
for ifile to num_snd

	## Open a sound file
	select Strings snd_list
	sound$ = Get string... ifile
	Read from file... 'snd_dir$''sound$'
	snd_name$ = selected$ ("Sound", 1)

	## Open a TextGrid file
	input_txt$ = "'txt_dir$''snd_name$''txt_ext$'"	
	Read from file... 'input_txt$'

	## Open Window to Edit TextGrid
	select Sound 'snd_name$'
	plus TextGrid 'snd_name$'
	Edit
	pause "Move on to the next file."
	
	## Save a modified TextGrid	
	out_name$ = "'txt_dir$''snd_name$''txt_ext$'"
	select TextGrid 'snd_name$'
	Write to text file... 'out_name$'
	
	## Remove objects from list
	select Sound 'snd_name$'
	plus TextGrid 'snd_name$'
	Remove
endfor
select all
Remove
	