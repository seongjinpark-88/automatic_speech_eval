require(Metrics)
require(ggplot2)

setwd("/Users/seongjinpark/PhD/Diss/automatic_speech_eval/keras_asa")
data = read.csv("./results/accented_massage.txt", sep = "\t")

head(data)
summary(data)

colnames(data) = c("score", "language", "stimuli", "wav")
summary(data)

eng_data = data[data$language == "ENG",]
View(eng_data)

sum_data = do.call(data.frame, aggregate(score ~ stimuli + language, data = eng_data, 
                                         function(x) c(mean = mean(x), sd = sd(x))))
View(sum_data)

# Threshold: score < 4 as a missing data (exclude listeners) 
high = sum_data[sum_data$score.mean >= 5.5 & sum_data$score.sd < 1, ]
View(high)
