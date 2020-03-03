## Praat Script to extract WAV files and its TextGrid from long WAV and TextGrid files.
## The name of output(the sound and the TextGrid file) will be same.
## 2015.01.13. Written by Seongjin
## 2016.06.28. Modified to extract sounds when its label is the same as the one previously assigned.
## 2019.04.22. Modified to extract all intervals which are not empty

form Autosegmentation sample
	comment File information
	sentence snd_dir ./wav/
	sentence snd_ext .wav
	sentence grid_dir ./TextGrid/
	sentence grid_ext .TextGrid
	sentence tier sentences
	comment Output information
	sentence out_dir ./tmp/
endform

system_nocheck mkdir "'out_dir$'"

## Create a list of all sound files
Create Strings as file list... list 'snd_dir$'*'snd_ext$'
num_snd = Get number of strings

## Main
for ifile to num_snd
	
	## Open a sound file
	select Strings list
	sound$ = Get string... ifile
	Open long sound file... 'snd_dir$''sound$'
	snd_name$ = selected$ ("LongSound", 1)
	
	## Open a TextGrid file
	Read from file... 'grid_dir$''snd_name$''grid_ext$'
	
	## Get index of tier with the name of the tier
	call GetTier 'tier$' tier

	## Get number of intervals in the tier
	tier_num = Get number of intervals... tier

	## number for wav file index
	number = 1
	for int to tier_num
		select TextGrid 'snd_name$'
		label$ = Get label of interval... tier int
		if label$ != ""
			# getting starting/ending point of interval
			start = Get starting point... tier int
			end = Get end point... tier int
			select LongSound 'snd_name$'
			Extract part... start end no
			# name of output file
			out_file$ = "'out_dir$''snd_name$'_'label$'.wav"
			if fileReadable (out_file$)
				pause 'out_file$' sound file exist. Continue?
			endif
			Write to WAV file... 'out_file$'
			Remove
			number = number + 1
		endif		
	endfor
endfor
select all
Remove

## Subroutine

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