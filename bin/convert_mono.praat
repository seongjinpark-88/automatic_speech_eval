## Praat Script for openning and modifying files easily.
## Open Sounds and TextGrids in same folder (with same name), and after modify, 
## you can move on to next files without saving
## 2015. 1. 13. Written by Seongjin

form Open files and Modify
	comment Sound information
	sentence snd_dir sounds/
	sentence snd_ext .wav
	sentence ext_dir mono/
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

	num_channels = Get number of channels

	if num_channels = 2
		new_sound = do ("Convert to mono")
		intervalfile$ = "'ext_dir$'" + "'snd_name$'" + ".wav"
		Write to WAV file... 'intervalfile$'
		Remove
	else
	  printline The file 'snd_name$' was alreday mono.
	endif

	## Remove objects from list
	select Sound 'snd_name$'
	Remove
endfor
select all
Remove
	