require(Metrics)
require(ggplot2)

setwd("/Users/seongjinpark/PhD/Diss/automatic_speech_eval/keras_asa")


###### ANALYZE CV RESUTS (FLUENCY) ######
mel_data = read.csv("./results/mel_10CV_fluency_wL1.txt", sep = "\t")
summary(mel_data)
colnames(mel_data) = c("CV", "Stimuli", "language", "true", "pred")
mel_data$CV = as.factor(mel_data$CV)
summary(mel_data)
View(mel_data)
mse(mel_data$true, mel_data$pred)
cor.test(mel_data$true, mel_data$pred)

eng_mel_data = mel_data[mel_data$language == "ENG", ]
kor_mel_data = mel_data[mel_data$language == "KOR", ]
chn_mel_data = mel_data[mel_data$language == "CHN", ]

mse(eng_mel_data$true, eng_mel_data$pred)
mse(kor_mel_data$true, kor_mel_data$pred)
mse(chn_mel_data$true, chn_mel_data$pred)

cor.test(eng_mel_data$true, eng_mel_data$pred)
cor.test(kor_mel_data$true, kor_mel_data$pred)
cor.test(chn_mel_data$true, chn_mel_data$pred)

ggplot(mel_data, aes(x = true, y = pred, color = language)) + stat_summary(geom="line") +
  ylim(-1, 6)


mfcc_data = read.csv("./results/mfcc_10CV_fluency_wL1.txt", sep = "\t")
summary(mfcc_data)
colnames(mfcc_data) = c("CV", "Stimuli", "language", "true", "pred")
mfcc_data$CV = as.factor(mfcc_data$CV)
summary(mfcc_data)

mse(mfcc_data$true, mfcc_data$pred)
cor.test(mfcc_data$true, mfcc_data$pred)

eng_mfcc_data = mfcc_data[mfcc_data$language == "ENG", ]
kor_mfcc_data = mfcc_data[mfcc_data$language == "KOR", ]
chn_mfcc_data = mfcc_data[mfcc_data$language == "CHN", ]

mse(eng_mfcc_data$true, eng_mfcc_data$pred)
mse(kor_mfcc_data$true, kor_mfcc_data$pred)
mse(chn_mfcc_data$true, chn_mfcc_data$pred)

cor.test(eng_mfcc_data$true, eng_mfcc_data$pred)
cor.test(kor_mfcc_data$true, kor_mfcc_data$pred)
cor.test(chn_mfcc_data$true, chn_mfcc_data$pred)

ggplot(mfcc_data, aes(x = true, y = pred, color = language)) + stat_summary()

##### ACCENTED DATA THRESHOLD ### 

data = read.csv("./results/accented_massage.txt", sep = "\t")
head(data)
summary(data)
colnames(data) = c("score", "language", "stimuli", "wav")
summary(data)
eng_data = data[data$language == "ENG",]
sum_data = do.call(data.frame, aggregate(score ~ stimuli + language, data = eng_data, 
                                         function(x) c(mean = mean(x), sd = sd(x))))
View(sum_data)
# Threshold: score < 4 as a missing data (exclude listeners) 
high = sum_data[sum_data$score.mean >= 5.5 & sum_data$score.sd < 1, ]
View(high)

#### FLUENCY DATA THRESHOLD ####
f_data = read.csv("./results/massaged_fluency.txt", sep = "\t")
summary(f_data)
colnames(f_data) = c("score","language","stimuli","sentence")
summary(f_data)
eng_f_data = f_data[f_data$language == "ENG",]
sum_f_data = do.call(data.frame, aggregate(score ~ stimuli + language, data = eng_f_data, 
                                         function(x) c(mean = mean(x), sd = sd(x))))
# Threshold: score < 4 as a missing data (exclude listeners) 
length(sum_f_data$score.mean)
high = sum_f_data[sum_f_data$score.mean >= 5.5 & sum_f_data$score.sd < 1, ]
View(high)



### COR between accent & fluency
fluency_result = read.csv("./results/mel_50_fluency_regression.txt", sep = "\t")
summary(fluency_result)
colnames(fluency_result) = c("stimuli", "true", "prediction")

accented_result = read.csv("./results/mel_40_accented_regression.txt", sep = "\t")
summary(accented_result)
colnames(accented_result) = c("stimuli", "true", "prediction")

summary(fluency_result)
summary(accented_result)

length(accented_result$true)
length(fluency_result$true)

accented_massaged = accented_result$true[1:length(fluency_result$true)]
cor.test(fluency_result$true, accented_massaged)

newdata = data.frame("fluency" = fluency_result$true, "accented" = accented_massaged)

ggplot(newdata, (aes(x = accented, y = fluency))) + geom_smooth(method="lm", color = "red")
       