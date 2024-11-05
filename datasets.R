#Conversão dos targets das bases do doutorado para +1 ou -1
#Data:05/11/2024

#Limpando o Ambiente global de variáveis
rm(list = ls())

#------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------
library(ggplot2)
library(dplyr)

#------------------------------------------------
#Carregando os datasets
#------------------------------------------------
#Diabetes
diabetes$Outcome = ifelse(diabetes$Outcome == 1, 1, -1)
unique(diabetes$Outcome)
table(diabetes$Outcome)/nrow(diabetes) * 100

#Exportando
write.csv(diabetes, "diabetes.csv", row.names = FALSE)

#Haberman
colnames(haberman) = c('V1', 'V2', 'V3', 'Class')
haberman$Class = ifelse(haberman$Class == 1, 1, -1)
unique(haberman$Class)
table(haberman$Class)/nrow(haberman) * 100

#Exportando
write.csv(haberman, "haberman.csv", row.names = FALSE)

#Hepatitis
hepatitis$X1.2 = ifelse(hepatitis$X1.2 == 1, 1, -1)
table(hepatitis$X1.2)/nrow(hepatitis) * 100

#Exportando
write.csv(hepatitis, "hepatitis.csv", row.names = FALSE)

#Ionosphere
ionosphere$V35 = ifelse(ionosphere$V35 == 'g', 1, -1)
table(ionosphere$V35)/nrow(ionosphere) * 100

#Exportando
write.csv(ionosphere, "ionosphere.csv", row.names = FALSE)

#Credit
SouthGermanCredit$kredit = ifelse(SouthGermanCredit$kredit == 1, 1, -1)
table(SouthGermanCredit$kredit)/nrow(SouthGermanCredit) * 100

#Exportando
write.csv(SouthGermanCredit, "SouthGermanCredit.csv", row.names = FALSE)

#Coluna
column_2C$V7 = ifelse(column_2C$V7 == "AB", 1, -1)
table(column_2C$V7)/nrow(column_2C) * 100

#Exportando
write.csv(column_2C, "column_2C.csv", row.names = FALSE)
