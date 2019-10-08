library(arules)
library(sqldf)
d <- read.table('G:/Mi unidad/Desarrollo/datasets/hipertension8var.csv', header = TRUE, sep = ',', dec = '.')
df <- sqldf("select * from d where Dx in (1,2)")
nombres <- c('weight', 'familySize', 'adviceSmoking', 'adviceSalt', 'age', 'cigarettesPerDay', 'classWorked','diabetes')
nombres <- as.vector(nombres)
for (i in 1:length(nombres))
  names(df)[i+1]<- nombres[i]

df$Dx <- as.factor(df$Dx)
df$familySize <- as.factor(df$familySize)
df$adviceSmoking <- as.factor(df$adviceSmoking)
df$adviceSalt <-  as.factor(df$adviceSalt)
df$classWorked <- as.factor(df$classWorked)
df$diabetes <- as.factor(df$diabetes)

intervalos <- cut(df$age, c(0, 18, 30, 45, 65, 100)) 
valores <- ordered(intervalos, labels = c("Menor", "Joven", "Medio", "Maduro", "Mayor")) 
df$age <- valores 

intervalos <- cut(df$cigarettesPerDay, c(-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) 
valores <- ordered(intervalos) 
df$cigarettesPerDay <- valores 

b <- trunc((max(df$weight) - min(df$weight))/15)
intervalos <- cut(df$weight, seq(min(df$weight) - 1, max(df$weight) + 3, by=b+1)) 
valores <- ordered(intervalos)
df$weight <- valores

# no funciona especificando rhs, revisar
#reglas <- apriori(df,parameter = list (supp = 0.7, conf= 0.9, target = "rules"), appearance = list(rhs = "Dx"))
reglas <- apriori(df,parameter = list (supp = 0.7, conf= 0.9, target = "rules"))
itemLabels(reglas)
inspect(sort(x = reglas, decreasing = TRUE, by = "confidence"))
b <- subset(x = reglas, subset = rhs %ain% "Dx=2")
inspect(sort(x = b, decreasing = TRUE, by = "confidence"))
