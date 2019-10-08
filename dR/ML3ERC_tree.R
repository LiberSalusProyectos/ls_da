library(tree)
df <- read.csv('G:/Mi Unidad/Desarrollo/datasets/kidney_disease_paraERC.csv')

df$classification[df$classification=='ckd\t'] <- 'ckd'
df$cad[df$cad=='\tno'] <- 'no'
df$dm[df$dm=='\tno'] <- 'no'
df$dm[df$dm=='\tyes'] <- 'yes'
df$dm[df$dm==' yes'] <- 'yes'
df <- df[,-1]
df <- na.omit(df)

train <- sample(1:nrow(df), size = nrow(df)*0.7)
setup <- tree.control(nobs = nrow(df[train,]), mincut = 15, minsize = 30, mindev = 0.01)
arbol_regresion <- tree(formula = classification ~ ., data = df, subset = train, split = "deviance", control = setup)
str(df)
